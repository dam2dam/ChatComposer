import React from "react";
import Chat from "./chat";
import InstrumentDropbox from "./instrumentDropbox";
import dynamic from "next/dynamic";

function Sequencer({
  trackId,
  setTrackIds,
}: // trackIds,
{
  trackId: number;
  setTrackIds: any;
  // trackIds: any[];
}) {
  const Transporter = dynamic(() => import("./transporter"), {
    ssr: false,
  });
  const DrumDropbox = dynamic(() => import("./drumDropbox"), {
    ssr: false,
  });
  return (
    <div className="text-gray-400">
      <div className="flex">
        {trackId === 3 ? (
          <div className="w-full border border-white">
            <DrumDropbox />
          </div>
        ) : (
          <div className="w-1/6 border border-white">
            <InstrumentDropbox trackId={trackId} />
          </div>
        )}

        {trackId === 3 ? (
          <></>
        ) : (
          <div className="w-full border border-white">
            <Chat
              trackId={trackId}
              setTrackIds={setTrackIds}
              // trackIds={trackIds}
            />
          </div>
        )}
      </div>
      {/* <div className="border border-white h-28">
        <Transporter trackId={trackId} />
      </div> */}
    </div>
  );
}

export default React.memo(Sequencer);
