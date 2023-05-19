import React from "react";
import { AlbumCoverType } from "../../../../types/musics";

function PlayListSong({ song }: { song: AlbumCoverType }) {
  const {
    cover_source: coverSource,
    member: { nickname },
    title,
  } = song;
  return (
    <div className="mp_playlist_song flex justify-between items-center transition-[0.2s] duration-[ease-in-out] px-0 py-[5px] hover:cursor-pointer hover:bg-[rgba(0,0,0,0.2)]">
      <div className="mp_tracks_album flex w-fit h-fit gap-2.5">
        <img
          src={coverSource}
          alt=""
          className="mp_tracks_album_img w-[60px]"
        />
        <div className="tracks">
          <h4 className="mp_tracks_album_h4 text-[15px] text-white">{title}</h4>
          <p className="mp_tracks_album_p text-[13px] text-[rgb(152,152,152)]">
            {nickname}
          </p>
        </div>
      </div>
      <div className="mp_tracks_menu">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAAXUlEQVRYhe3UsRGAMAwEwXuKgGZMaSZwbSoGmrDoQePMv7k0Hx2Y7U7Vw4i4M/MBUtJorUXlz1EdkJkdOIEL6NU/5QGrlAdIGsAn6Z1zjoWbbDPugDtg5g64A2b2AwjvOQGxKmSGAAAAAElFTkSuQmCC" />
      </div>
    </div>
  );
}

export default PlayListSong;
