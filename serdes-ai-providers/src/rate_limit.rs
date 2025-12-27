//! Rate limiting utilities.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::Mutex;

/// Token bucket rate limiter.
pub struct RateLimiter {
    tokens: AtomicU64,
    max_tokens: u64,
    refill_rate: f64,
    last_refill: Mutex<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    #[must_use]
    pub fn new(max_tokens: u64, refill_per_second: f64) -> Self {
        Self {
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            refill_rate: refill_per_second,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Try to acquire a token. Returns true if successful.
    pub fn try_acquire(&self) -> bool {
        self.refill();
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false;
            }
            if self.tokens.compare_exchange_weak(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Wait until a token is available.
    pub async fn acquire(&self) {
        while !self.try_acquire() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn refill(&self) {
        let mut last = self.last_refill.lock();
        let elapsed = last.elapsed().as_secs_f64();
        let new_tokens = (elapsed * self.refill_rate) as u64;
        if new_tokens > 0 {
            *last = Instant::now();
            let current = self.tokens.load(Ordering::Relaxed);
            let new_value = (current + new_tokens).min(self.max_tokens);
            self.tokens.store(new_value, Ordering::Relaxed);
        }
    }
}
