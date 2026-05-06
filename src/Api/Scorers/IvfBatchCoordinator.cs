using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace Rinha.Api.Scorers;

/// <summary>
/// L1: lightweight batcher in front of <see cref="IvfScorer"/>. Submitted queries land
/// in a channel; a single worker pulls them, opportunistically pairs two arrivals that
/// land within a short time window, and runs <see cref="IvfScorer.ScoreBatch2"/> on the
/// pair. Single arrivals fall back to <see cref="IvfScorer.Score"/>.
///
/// Trade-off: adds a small wait (~50µs) hoping to pair up so the next-row Q8 read is
/// amortized across two queries. Net win iff incoming RPS is high enough that pairs
/// form often AND the joint scan is faster than 2x sequential scans.
/// </summary>
public sealed class IvfBatchCoordinator : IDisposable
{
    private readonly IvfScorer _scorer;
    private readonly Channel<QueryItem> _channel;
    private readonly Thread[] _workers;
    private readonly int _pairWaitTicks;
    private volatile bool _stopping;

    public IvfBatchCoordinator(IvfScorer scorer, int pairWaitMicros = 50, int workerCount = 2)
    {
        _scorer = scorer;
        _channel = Channel.CreateUnbounded<QueryItem>(new UnboundedChannelOptions
        {
            SingleReader = false,
            SingleWriter = false,
            AllowSynchronousContinuations = false,
        });
        _pairWaitTicks = (int)((long)pairWaitMicros * Stopwatch.Frequency / 1_000_000L);
        _workers = new Thread[Math.Max(1, workerCount)];
        for (int i = 0; i < _workers.Length; i++)
        {
            _workers[i] = new Thread(Run)
            {
                IsBackground = true,
                Name = $"ivf-batch-worker-{i}",
                Priority = ThreadPriority.AboveNormal,
            };
            _workers[i].Start();
        }
    }

    public Task<float> SubmitAsync(float[] query)
    {
        var item = new QueryItem(query);
        if (!_channel.Writer.TryWrite(item))
        {
            // Unbounded channel never refuses; this branch is dead but keeps the API contract.
            return Task.FromResult(_scorer.Score(query));
        }
        return item.Tcs.Task;
    }

    private void Run()
    {
        var reader = _channel.Reader;
        while (!_stopping)
        {
            QueryItem first;
            try
            {
                if (!reader.WaitToReadAsync().AsTask().GetAwaiter().GetResult()) return;
            }
            catch { return; }
            if (!reader.TryRead(out first!)) continue;

            // Try to pair with a second arrival within _pairWaitTicks.
            QueryItem? second = null;
            if (!reader.TryRead(out second))
            {
                long deadline = Stopwatch.GetTimestamp() + _pairWaitTicks;
                while (Stopwatch.GetTimestamp() < deadline)
                {
                    if (reader.TryRead(out second)) break;
                    Thread.SpinWait(40);
                }
            }

            try
            {
                if (second is null)
                {
                    float s = _scorer.Score(first.Query);
                    first.Tcs.SetResult(s);
                }
                else
                {
                    _scorer.ScoreBatch2(first.Query, second.Query, out float s1, out float s2);
                    first.Tcs.SetResult(s1);
                    second.Tcs.SetResult(s2);
                }
            }
            catch (Exception ex)
            {
                first.Tcs.TrySetException(ex);
                second?.Tcs.TrySetException(ex);
            }
        }
    }

    public void Dispose()
    {
        _stopping = true;
        _channel.Writer.TryComplete();
    }

    private sealed class QueryItem
    {
        public readonly float[] Query;
        public readonly TaskCompletionSource<float> Tcs;
        public QueryItem(float[] q)
        {
            Query = q;
            Tcs = new TaskCompletionSource<float>(TaskCreationOptions.RunContinuationsAsynchronously);
        }
    }
}
