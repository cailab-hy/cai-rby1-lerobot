// Captures microphone PCM in fixed frames and reports each frame's loudness.
// The main thread runs the voice-activity state machine and uploads whole
// utterances, so this processor stays deliberately dumb.
const FRAME_SAMPLES = 512; // 32 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(FRAME_SAMPLES);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let index = 0; index < channel.length; index += 1) {
      this.frame[this.offset] = channel[index];
      this.offset += 1;
      if (this.offset < FRAME_SAMPLES) continue;

      let energy = 0;
      for (let sample = 0; sample < FRAME_SAMPLES; sample += 1) {
        energy += this.frame[sample] * this.frame[sample];
      }
      const samples = this.frame.slice();
      this.port.postMessage(
        { rms: Math.sqrt(energy / FRAME_SAMPLES), samples },
        [samples.buffer],
      );
      this.offset = 0;
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
