//! Temporal grouping for efficient streaming.
//!
//! This module provides utilities for debouncing and grouping stream events
//! to reduce overhead in high-frequency streaming scenarios.

use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

pin_project! {
    /// Groups streaming events by time to reduce overhead.
    ///
    /// This stream buffers incoming items and yields them in batches
    /// after a specified debounce interval.
    pub struct DebouncedStream<S>
    where
        S: Stream,
    {
        #[pin]
        inner: S,
        debounce_interval: Duration,
        buffer: Vec<S::Item>,
        last_emit: Option<Instant>,
        finished: bool,
    }
}

impl<S> DebouncedStream<S>
where
    S: Stream,
{
    /// Create a new debounced stream.
    pub fn new(inner: S, debounce: Duration) -> Self {
        Self {
            inner,
            debounce_interval: debounce,
            buffer: Vec::new(),
            last_emit: None,
            finished: false,
        }
    }

    /// Get the debounce interval.
    pub fn interval(&self) -> Duration {
        self.debounce_interval
    }

    /// Get the current buffer size.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

impl<S> Stream for DebouncedStream<S>
where
    S: Stream + Unpin,
    S::Item: Clone,
{
    type Item = Vec<S::Item>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.finished && this.buffer.is_empty() {
            return Poll::Ready(None);
        }

        // Poll inner stream until pending or done
        loop {
            match this.inner.poll_next_unpin(cx) {
                Poll::Ready(Some(item)) => {
                    this.buffer.push(item);

                    // Check if we should emit
                    let should_emit = match this.last_emit {
                        Some(last) => last.elapsed() >= *this.debounce_interval,
                        None => false,
                    };

                    if should_emit && !this.buffer.is_empty() {
                        *this.last_emit = Some(Instant::now());
                        let batch = std::mem::take(this.buffer);
                        return Poll::Ready(Some(batch));
                    }
                }
                Poll::Ready(None) => {
                    *this.finished = true;
                    // Emit remaining buffer
                    if !this.buffer.is_empty() {
                        let batch = std::mem::take(this.buffer);
                        return Poll::Ready(Some(batch));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    // If we have buffered items and debounce time passed, emit
                    let should_emit = match this.last_emit {
                        Some(last) => last.elapsed() >= *this.debounce_interval,
                        None => !this.buffer.is_empty(),
                    };

                    if should_emit && !this.buffer.is_empty() {
                        *this.last_emit = Some(Instant::now());
                        let batch = std::mem::take(this.buffer);
                        return Poll::Ready(Some(batch));
                    }

                    return Poll::Pending;
                }
            }
        }
    }
}

pin_project! {
    /// Stream that throttles items to a maximum rate.
    pub struct ThrottledStream<S>
    where
        S: Stream,
    {
        #[pin]
        inner: S,
        min_interval: Duration,
        last_emit: Option<Instant>,
        pending_item: Option<S::Item>,
    }
}

impl<S> ThrottledStream<S>
where
    S: Stream,
{
    /// Create a new throttled stream.
    pub fn new(inner: S, min_interval: Duration) -> Self {
        Self {
            inner,
            min_interval,
            last_emit: None,
            pending_item: None,
        }
    }
}

impl<S> Stream for ThrottledStream<S>
where
    S: Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Check if we have a pending item
        if let Some(item) = this.pending_item.take() {
            let can_emit = match this.last_emit {
                Some(last) => last.elapsed() >= *this.min_interval,
                None => true,
            };

            if can_emit {
                *this.last_emit = Some(Instant::now());
                return Poll::Ready(Some(item));
            } else {
                *this.pending_item = Some(item);
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        }

        // Poll inner stream
        match this.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(item)) => {
                let can_emit = match this.last_emit {
                    Some(last) => last.elapsed() >= *this.min_interval,
                    None => true,
                };

                if can_emit {
                    *this.last_emit = Some(Instant::now());
                    Poll::Ready(Some(item))
                } else {
                    *this.pending_item = Some(item);
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

pin_project! {
    /// Stream that coalesces multiple text items into larger chunks.
    pub struct CoalescedTextStream<S> {
        #[pin]
        inner: S,
        buffer: String,
        min_chunk_size: usize,
        max_chunk_size: usize,
        finished: bool,
    }
}

impl<S> CoalescedTextStream<S>
where
    S: Stream<Item = String>,
{
    /// Create a new coalesced text stream.
    pub fn new(inner: S, min_chunk_size: usize, max_chunk_size: usize) -> Self {
        Self {
            inner,
            buffer: String::new(),
            min_chunk_size,
            max_chunk_size,
            finished: false,
        }
    }
}

impl<S> Stream for CoalescedTextStream<S>
where
    S: Stream<Item = String> + Unpin,
{
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.finished && this.buffer.is_empty() {
            return Poll::Ready(None);
        }

        loop {
            // Check if buffer exceeds max size
            if this.buffer.len() >= *this.max_chunk_size {
                let chunk = std::mem::take(this.buffer);
                return Poll::Ready(Some(chunk));
            }

            match this.inner.poll_next_unpin(cx) {
                Poll::Ready(Some(text)) => {
                    this.buffer.push_str(&text);

                    // Emit if we've reached max size
                    if this.buffer.len() >= *this.max_chunk_size {
                        let chunk = std::mem::take(this.buffer);
                        return Poll::Ready(Some(chunk));
                    }
                }
                Poll::Ready(None) => {
                    *this.finished = true;
                    if !this.buffer.is_empty() {
                        let chunk = std::mem::take(this.buffer);
                        return Poll::Ready(Some(chunk));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    // Emit buffer if it meets minimum size
                    if this.buffer.len() >= *this.min_chunk_size {
                        let chunk = std::mem::take(this.buffer);
                        return Poll::Ready(Some(chunk));
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

/// Extension trait for adding debouncing capabilities to streams.
pub trait StreamDebounceExt: Stream {
    /// Debounce the stream, grouping items by time.
    fn debounce(self, duration: Duration) -> DebouncedStream<Self>
    where
        Self: Sized,
    {
        DebouncedStream::new(self, duration)
    }

    /// Throttle the stream to a maximum rate.
    fn throttle(self, min_interval: Duration) -> ThrottledStream<Self>
    where
        Self: Sized,
    {
        ThrottledStream::new(self, min_interval)
    }
}

impl<S: Stream> StreamDebounceExt for S {}

/// Extension trait for text streams.
pub trait TextStreamExt: Stream<Item = String> {
    /// Coalesce text items into larger chunks.
    fn coalesce(self, min_size: usize, max_size: usize) -> CoalescedTextStream<Self>
    where
        Self: Sized,
    {
        CoalescedTextStream::new(self, min_size, max_size)
    }
}

impl<S: Stream<Item = String>> TextStreamExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_debounced_stream() {
        let items = vec![1, 2, 3, 4, 5];
        let inner = stream::iter(items);
        let debounced = DebouncedStream::new(inner, Duration::from_millis(10));

        let batches: Vec<Vec<i32>> = debounced.collect().await;

        // All items should be in batches
        let total: i32 = batches.iter().flat_map(|b| b.iter()).sum();
        assert_eq!(total, 15);
    }

    #[tokio::test]
    async fn test_throttled_stream() {
        let items = vec![1, 2, 3];
        let inner = stream::iter(items);
        let throttled = ThrottledStream::new(inner, Duration::from_millis(1));

        let results: Vec<i32> = throttled.collect().await;
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_coalesced_text_stream() {
        let items = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let inner = stream::iter(items);
        let coalesced = CoalescedTextStream::new(inner, 2, 10);

        let results: Vec<String> = coalesced.collect().await;

        // Items should be coalesced
        let total_len: usize = results.iter().map(|s| s.len()).sum();
        assert_eq!(total_len, 4);
    }

    #[tokio::test]
    async fn test_extension_traits() {
        let items = vec![1, 2, 3];
        let inner = stream::iter(items);

        let results: Vec<Vec<i32>> = inner.debounce(Duration::from_millis(1)).collect().await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_text_extension() {
        let items = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
        let inner = stream::iter(items);

        let results: Vec<String> = inner.coalesce(5, 100).collect().await;
        assert!(!results.is_empty());
    }
}
