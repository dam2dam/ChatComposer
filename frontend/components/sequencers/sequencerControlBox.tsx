import { IconPlayerPlayFilled } from "@tabler/icons-react";
import React from "react";

function SequencerControlBox({ onPlay }) {
  return (
    <>
      <div role="button" className="text-pink-500" onClick={onPlay}>
        <IconPlayerPlayFilled strokeWidth={2} color={"pink"} />
      </div>
    </>
  );
}

export default SequencerControlBox;
