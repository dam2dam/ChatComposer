import React from "react";
import { SongsDummyData } from "../../library/favorite/favoriteSong";

function Player() {
  const song = SongsDummyData[3];
  const {
    title,
    member: { nickname },
    cover_source: coverSource,
  } = song;

  return (
    <div className="mp_playlist_player absolute h-60 w-[90%] shadow-[0_0_20px_rgb(0,0,0)] transition-[0.2s] duration-[ease-in-out] rounded-[10px] left-[17px] bottom-5 bg-black/20 hover:cursor-pointer  hover:scale-[1.02]">
      <img
        src={coverSource}
        alt=""
        className="mp_playlist_player_img relative w-full h-full rounded-[10px]"
      />
      <div className="mp_playlist_controller absolute w-full h-3/6 bg-[#0000007b] backdrop-blur-[10px] flex flex-col items-center justify-center gap-[15px] rounded-[0_0_10px_10px] left-0 bottom-0">
        <div className="mp_playlist_control_song flex flex-col justify-center items-center">
          <h4 className="mp_playlist_control_song_h4 text-xl text-white">
            {title}
          </h4>
          <p className="mp_playlist_control_song_p text-[15px] text-[rgb(152,152,152)]">
            {nickname}
          </p>
        </div>
        <div className="mp_playlist_control flex justify-center items-center gap-10">
          <img
            src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAAgklEQVRIie3TsQ3CMBgF4QilJhU7ZAWWYRZmicQA7MEIjEGJ9NFQWL9sEWJM5avvnSU7GYZO5y9gj6XWKQ2PuEONkxuNOOPpzRanFJ9xE/jWKcVPeMRhOl7j5MIHXHOjhI9O7O5W3dkvaXZF4ZB2j5wE2n2mIdLmRwuBCZdap9PZxgtDc55Ko9iBRQAAAABJRU5ErkJggg=="
            className="mp_playlist_control_img w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125"
          />
          <img
            src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAABmJLR0QA/wD/AP+gvaeTAAAB5ElEQVRoge2ZMS8EQRTH7wgSIiEhcY1GcxoUdDqViE8g8RVUeo0PoFRSalWicRrR6BRoFBQkCiIhuPsp7m3y7m7Pzd3O7pjJ/prL3s689/9nZ3fezBQKOTk5ORpgGxh0rSMx1LkBVl1rSQSNnAKzrjX1hDLxJr+fwC4w4lpbVygjJWAfqMr1I7AJFF1rNCJyoa4XgQtlsALMu9RoRLMR+a9PnsaT3K4Ch8CkK50diTOi7o0Be8C3NHsBtoD+rHV25C8jqk0ZOFHD7QpYzkqjESZGVNt14F4ZOgam09ZoRDdGpP0wsAN8SNd3uR5KU6eJsK6MqH4zwJF6OnfAWhoaTQX1ZET1XwGulSE31UFSIxJjQL5mrxLuS752o7Z0mohIbETFKlGfb2oSNrvqwKYRFXMJuFTDrQLM2cwRl9S6EYkbVQfPkiKqDiZs54oSpmJExR+X9+VHUqVTHaRtROVZAM7VcLNbHWRlRHIVgQ3gQdLWgANgql2fviyE/SvyoWUe3++XPYjPL60T4hk+TYj4XqIQQtFIfBlftqnRVEhPRmhdWN3i08KKEJa6NG4+1OSJ+LP5gO/bQfi+QUf7LdN0ZmUbNBshfhM73VnZBkpwGMcKBHTQAwEcvQVxGBrG8XROTo7f/AKcSk4xvp2PUQAAAABJRU5ErkJggg=="
            id="play_pause"
            className="mp_playlist_control_img w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125"
          />
          <img
            src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAAdklEQVRIie3TuxGDMBBFUeyAlBlC90APdEQtFOHEHdAGHbgJhxwipbZkSUOiG9/dN7Ofrms0ioMXxlznWzG8Mec4vwLgwIr+HycmILBjSnVSAuCDBbdYJzUgsOER44R+9+jZleKKEVVbctUzrfpoTwy5TqNRjhM3UKtw78EvwwAAAABJRU5ErkJggg=="
            className="mp_playlist_control_img w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125"
          />
        </div>
      </div>
    </div>
  );
}

export default Player;
