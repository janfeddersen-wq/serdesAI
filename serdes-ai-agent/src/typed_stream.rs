//! Typed final output without changing existing stream event types.
use crate::{AgentRunError, AgentStream, AgentStreamEvent};
use futures::{Stream, StreamExt};
use std::{
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};
/// A compatible event stream with a recoverable validator-transformed result.
/// Output becomes available only after the successful terminal checkpoint.
pub struct TypedAgentStream<Output> {
    pub(crate) stream: AgentStream,
    pub(crate) marker: PhantomData<fn() -> Output>,
}
impl<O: Send + Sync + 'static> TypedAgentStream<O> {
    /// Take validated output once; None before terminal commit or for partial runs.
    pub fn take_output(&mut self) -> Option<O> {
        self.stream.take_typed_output()
    }
    /// Drain events with backpressure, returning the exact transformed value.
    /// A valid token-limit/content-filter partial has no typed final output.
    pub async fn finish(mut self) -> Result<Option<O>, AgentRunError> {
        while let Some(event) = self.next().await {
            event?;
        }
        Ok(self.take_output())
    }
    /// Cancel this run.
    pub fn cancel(&self) {
        self.stream.cancel();
    }
}
impl<O> Stream for TypedAgentStream<O> {
    type Item = Result<AgentStreamEvent, AgentRunError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().stream).poll_next(cx)
    }
}
