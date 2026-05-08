//! Connection / runtime configuration for an [`OllamaTransport`](super::transport::OllamaTransport).
//!
//! The defaults match Ollama's out-of-the-box install (`http://localhost:11434`)
//! and a generous request timeout suitable for local generation. The
//! `loopback_only` flag is on by default to keep the very first version of
//! this integration safe by construction; opt-in is required to point Warp
//! at a remote Ollama host.

use std::time::Duration;

use url::Url;

use super::error::OllamaError;

/// Default Ollama endpoint as advertised by `ollama serve`.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// Default per-request timeout. Local models can be slow to first-token; the
/// per-token streaming timeout is enforced separately by the transport.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL of the Ollama daemon, e.g. `http://localhost:11434`.
    pub base_url: Url,
    /// Hard cap on a single HTTP request (excluding streaming idle time).
    pub request_timeout: Duration,
    /// Forwarded as the `keep_alive` parameter on `/api/chat`. `None` lets
    /// Ollama use its own default (5 minutes at the time of writing).
    pub keep_alive: Option<Duration>,
    /// When `true`, [`OllamaConfig::validate`] rejects any non-loopback host.
    /// This is the default and SHOULD only be relaxed when the user has
    /// explicitly opted into a remote Ollama host in settings.
    pub loopback_only: bool,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: Url::parse(DEFAULT_OLLAMA_BASE_URL)
                .expect("DEFAULT_OLLAMA_BASE_URL must parse"),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            keep_alive: None,
            loopback_only: true,
        }
    }
}

impl OllamaConfig {
    /// Build a config from a user-supplied base URL string, applying the
    /// security constraints in [`OllamaConfig::validate`].
    pub fn from_base_url(raw: &str) -> Result<Self, OllamaError> {
        let mut cfg = Self::default();
        cfg.base_url = Url::parse(raw).map_err(|e| OllamaError::InvalidConfig {
            reason: format!("base URL `{raw}` is not a valid URL: {e}"),
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Returns `Ok(())` iff the base URL is one Warp is willing to talk to:
    /// scheme is `http`/`https`, no embedded credentials, and (when
    /// `loopback_only` is set) the host resolves syntactically to a loopback
    /// address. We deliberately do not perform DNS here — DNS-time checks
    /// would create a TOCTOU gap, and the loopback restriction is meant as
    /// a *safety rail*, not a security boundary.
    pub fn validate(&self) -> Result<(), OllamaError> {
        match self.base_url.scheme() {
            "http" | "https" => {}
            other => {
                return Err(OllamaError::InvalidConfig {
                    reason: format!("unsupported URL scheme `{other}`; use http or https"),
                });
            }
        }

        if !self.base_url.username().is_empty() || self.base_url.password().is_some() {
            return Err(OllamaError::InvalidConfig {
                reason: "embedded credentials in the base URL are not allowed".to_string(),
            });
        }

        if self.loopback_only && !is_loopback(&self.base_url) {
            return Err(OllamaError::InvalidConfig {
                reason: format!(
                    "host `{}` is not a loopback address; enable \
                     `Allow remote Ollama hosts` in settings to use it",
                    self.base_url.host_str().unwrap_or("<missing>")
                ),
            });
        }

        Ok(())
    }

    /// Joins a relative path (e.g. `"api/tags"`) against the base URL.
    pub fn endpoint(&self, path: &str) -> Result<Url, OllamaError> {
        // `Url::join` requires the base to end with `/` to behave intuitively.
        let mut base = self.base_url.clone();
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        base.join(path).map_err(|e| OllamaError::InvalidConfig {
            reason: format!("could not build endpoint for `{path}`: {e}"),
        })
    }
}

fn is_loopback(url: &Url) -> bool {
    use url::Host;
    match url.host() {
        Some(Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(addr)) => addr.is_loopback(),
        Some(Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_loopback() {
        let cfg = OllamaConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_non_loopback_when_locked() {
        let cfg = OllamaConfig::from_base_url("http://example.com:11434");
        assert!(matches!(cfg, Err(OllamaError::InvalidConfig { .. })));
    }

    #[test]
    fn allows_remote_host_when_opted_in() {
        let mut cfg = OllamaConfig::default();
        cfg.base_url = Url::parse("http://10.0.0.5:11434").unwrap();
        cfg.loopback_only = false;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_unsupported_scheme() {
        let cfg = OllamaConfig::from_base_url("file:///tmp/ollama");
        assert!(matches!(cfg, Err(OllamaError::InvalidConfig { .. })));
    }

    #[test]
    fn rejects_embedded_credentials() {
        let mut cfg = OllamaConfig::default();
        cfg.base_url = Url::parse("http://user:pass@localhost:11434").unwrap();
        assert!(matches!(cfg.validate(), Err(OllamaError::InvalidConfig { .. })));
    }

    #[test]
    fn accepts_ipv6_loopback() {
        let mut cfg = OllamaConfig::default();
        cfg.base_url = Url::parse("http://[::1]:11434").unwrap();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn endpoint_joins_paths() {
        let cfg = OllamaConfig::default();
        let url = cfg.endpoint("api/tags").unwrap();
        assert_eq!(url.as_str(), "http://localhost:11434/api/tags");
    }
}
