import React from "react";
import * as Tone from "tone";

function AudioTest() {
  const synth = new Tone.Synth().toDestination();
  const now = Tone.now();
  // trigger the attack immediately
  synth.triggerAttack("C4", now);
  // wait one second before triggering the release
  synth.triggerRelease(now + 1);
  return <div>AudioTest</div>;
}

export default AudioTest;
