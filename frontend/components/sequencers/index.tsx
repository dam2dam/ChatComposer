import React from "react";
import Sequencer from "./sequencer";
import PlayAllButton from "./playAllButton";
import ContentContainer from "../dashboard/library/contentContainer";

function Sequencers({ setTrackIds, trackIds }) {
  return (
    <div className="pt-10">
      <h3 className="w-full text-center libray_h3 font-bold text-pink-500 text-2xl mb-[15px] mt-2">
        작곡하기
      </h3>
      <ContentContainer>
        {trackIds.map((e, i) => {
          return (
            <Sequencer
              key={i}
              trackId={i}
              setTrackIds={setTrackIds}
              // trackIds={trackIds}
            />
          );
        })}
        <PlayAllButton />
      </ContentContainer>
    </div>
  );
}

export default Sequencers;
