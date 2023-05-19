import React, { useState } from "react";
import PianoChat from "./pianoChat";
import dynamic from "next/dynamic";

function PianoSequencer() {
  const [pianoNotes, setPianoNotes] = useState([]);
  const PianoTransport = dynamic(() => import("./pianoTransport"), {
    ssr: false,
  });
  return (
    <div>
      <PianoChat setNotes={setPianoNotes} />
      {pianoNotes.length ? <PianoTransport notes={pianoNotes} /> : <></>}
    </div>
  );
}

export default PianoSequencer;
