export class WebCam {
  config = undefined;
  element = undefined;
  stream = undefined;
  canvas = undefined;
  devices = undefined;

  constructor() {
    this.config = {
      element: undefined,
      debug: true,
      mode: 'front',
      crop: false,
      width: 0,
      height: 0,
    };
  }

  get track() {
    if (!this.stream) return undefined;
    return this.stream.getVideoTracks()[0];
  }

  get capabilities() {
    if (!this.track) return undefined;
    return this.track.getCapabilities ? this.track.getCapabilities() : undefined;
  }

  get constraints() {
    if (!this.track) return undefined;
    return this.track.getConstraints ? this.track.getConstraints() : undefined;
  }

  get settings() {
    if (!this.stream) return undefined;
    const track = this.stream.getVideoTracks()[0];
    return track.getSettings ? track.getSettings() : undefined;
  }

  get label() {
    if (!this.track) return '';
    return this.track.label;
  }

  get paused() {
    return this.element?.paused || false;
  }

  get width() {
    return this.element?.videoWidth || 0;
  }

  get height() {
    return this.element?.videoHeight || 0;
  }

  enumerate = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.devices = devices.filter((device) => device.kind === 'videoinput');
    } catch {
      this.devices = [];
    }
    return this.devices;
  };

  /** start method initializizes webcam stream and associates it with a dom video element */
  start = async (webcamConfig) => {
    // set config
    if (webcamConfig?.debug) this.config.debug = webcamConfig?.debug;
    if (webcamConfig?.crop) this.config.crop = webcamConfig?.crop;
    if (webcamConfig?.mode) this.config.mode = webcamConfig?.mode;
    if (webcamConfig?.width) this.config.width = webcamConfig?.width;
    if (webcamConfig?.height) this.config.height = webcamConfig?.height;
    if (webcamConfig?.id) this.config.id = webcamConfig?.id;

    if (webcamConfig?.canvas) {
      if (typeof webcamConfig.canvas === 'string') {
        const el = document.getElementById(webcamConfig.canvas);
        if (el && el instanceof HTMLCanvasElement) {
          this.canvas = el;
        } else {
          if (this.config.debug) log('webcam', 'cannot get dom element', webcamConfig.canvas);
        }
      } else if (webcamConfig.element instanceof HTMLCanvasElement) {
        this.canvas = webcamConfig.canvas;
      } else {
        if (this.config.debug) log('webcam', 'unknown dom element', webcamConfig.canvas);
        return `webcam error: unknown dom element: ${webcamConfig.canvas}`;
      }
    }

    // use or create dom element
    if (webcamConfig?.element) {
      if (typeof webcamConfig.element === 'string') {
        const el = document.getElementById(webcamConfig.element);
        if (el && el instanceof HTMLVideoElement) {
          this.element = el;
        } else {
          if (this.config.debug) log('webcam', 'cannot get dom element', webcamConfig.element);
          return `webcam error: cannot get dom element: ${webcamConfig.element}`;
        }
      } else if (webcamConfig.element instanceof HTMLVideoElement) {
        this.element = webcamConfig.element;
      } else {
        if (this.config.debug) log('webcam', 'unknown dom element', webcamConfig.element);
        return `webcam error: unknown dom element: ${webcamConfig.element}`;
      }
    } else {
      this.element = document.createElement('video');
    }

    // set constraints to use
    const requestedConstraints = {
      audio: false,
      video: {
        facingMode: this.config.mode === 'front' ? 'user' : 'environment',
        // @ts-ignore // resizeMode is still not defined in tslib
        resizeMode: this.config.crop ? 'crop-and-scale' : 'none',
      },
    };
    if (this.config?.width > 0) (requestedConstraints.video).width = { ideal: this.config.width };
    if (this.config?.height > 0) (requestedConstraints.video).height = { ideal: this.config.height };
    if (this.config.id) (requestedConstraints.video).deviceId = this.config.id;

    // set default event listeners
    this.element.addEventListener('play', () => { if (this.config.debug) log('webcam', 'play'); });
    this.element.addEventListener('pause', () => { if (this.config.debug) log('webcam', 'pause'); });
    this.element.addEventListener('click', async () => { // pause when clicked on screen and resume on next click
      if (!this.element || !this.stream) return;
      if (this.element.paused) await this.element.play();
      else this.element.pause();
    });

    // get webcam and set it to run in dom element
    if (!navigator?.mediaDevices) {
      if (this.config.debug) log('webcam error', 'no devices');
      return 'webcam error: no devices';
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia(requestedConstraints); // get stream that satisfies constraints
    } catch (err) {
      log('webcam', err);
      return `webcam error: ${err}`;
    }
    if (!this.stream) {
      if (this.config.debug) log('webcam error', 'no stream');
      return 'webcam error no stream';
    }
    this.element.srcObject = this.stream; // assign it to dom element
    const ready = new Promise((resolve) => { // wait until stream is ready
      if (!this.element) resolve(false);
      else this.element.onloadeddata = () => resolve(true);
    });
    await ready;

    await this.element.play(); // start playing

    if (this.config.debug) {
      log('webcam', {
        width: this.width,
        height: this.height,
        label: this.label,
        stream: this.stream,
        track: this.track,
        settings: this.settings,
        constraints: this.constraints,
        capabilities: this.capabilities,
      });
    }
    return `webcam: ${this.label}`;
  };

  pause = () => {
    if (this.element) this.element.pause();
  };

  play = async () => {
    if (this.element) await this.element.play();
    this.element.width = this.width;
    this.element.height = this.height;
  };

  stop = () => {
    if (this.config.debug) log('webcam', 'stop');
    if (this.track) this.track.stop();
  };
}
