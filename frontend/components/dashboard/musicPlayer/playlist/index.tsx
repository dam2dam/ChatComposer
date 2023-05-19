import React, { Fragment } from "react";
import { SongsDummyData } from "../../library/favorite/favoriteSong";
import PlayListSong from "./playListSong";

function Playlist() {
  return (
    <div className="mp_playlist_content absolute w-full h-3/5 flex flex-col gap-[15px] px-5 py-0 top-[16%] overflow-y-scroll custom_scrollbar">
      {SongsDummyData.map((song, index) => {
        return (
          <Fragment key={`playlist_content_${song.music_id}`}>
            {index ? (
              <hr className=".mp_playlist_content_hr h-[0.5px] w-full -mt-2 border-[none] bg-gray-500/40" />
            ) : (
              <></>
            )}
            <PlayListSong song={song} />
          </Fragment>
        );
      })}
    </div>
  );
}

export default Playlist;
