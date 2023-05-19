import React from "react";
import AudioVisualizer from "../../../src/components/machine/AudioVisualizer";

function Tracks() {
  return (
    <div className="mp_playlist absolute w-[350px] h-full rounded-[0_20px_20px_0] right-0 top-0 bg-black/20">
      <div className="absolute text-white text-[17px] left-5 top-[4%]">
        + 새로운 트랙 만드는 버튼(채팅창감)
      </div>
      <h3 className="mp_playlist_h3 absolute text-white text-[17px] left-5 top-[10%]">
        Tracks
      </h3>
      <div className="absolute left-5 top-[15%] 트랙목록임. 트랙마다 AudioVisualizer보여줄라고">
        <div className="border p-2 트랙 하나임">
          <div>트랙이름</div>
          <div className="w-60">
            <AudioVisualizer height={100} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Tracks;
