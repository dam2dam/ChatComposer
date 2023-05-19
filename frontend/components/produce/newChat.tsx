import React from "react";
import Chat from "../sequencers/chat";

function NewChat({ setTrackIds }) {
  return (
    <div>
      NewChat
      <Chat trackId={0} setTrackIds={setTrackIds} />;
    </div>
  );
}

export default NewChat;
