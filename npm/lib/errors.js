class PlatformNotSupportedError extends Error {
  constructor(message) {
    super(message);
    this.name = "PlatformNotSupportedError";
  }
}

module.exports = { PlatformNotSupportedError };
