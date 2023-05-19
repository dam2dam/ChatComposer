import { IconPlus } from "@tabler/icons-react";
import React from "react";

function TrackAdder({ setTrackIds, trackIds }) {
  return (
    <div
      className={`w-full bg-pink-400 hover:bg-pink-500 mt-2 flex ${
        trackIds.length >= 4 ? "hidden" : ""
      }`}
      role="button"
      onClick={() => setTrackIds((prev) => [...prev, ""])}
    >
      <div className="text-white flex text-center mx-auto">
        <IconPlus></IconPlus>
        트랙 추가
      </div>
    </div>
  );
}

export default TrackAdder;
