import React from "react";
import Playlist from "./playlist";
import Player from "./player";

function MusicPlayer() {
  return (
    <div className="mp_playlist absolute w-[350px] h-full rounded-[0_20px_20px_0] right-0 top-0 bg-black/20">
      <h3 className="mp_playlist_h3 absolute text-white text-[17px] left-5 top-[10%]">
        Current Playlist
      </h3>
      <Playlist />
      <Player />
    </div>
  );
}

export default MusicPlayer;
