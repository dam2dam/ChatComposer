import React, { useState, useRef, useEffect } from "react";
import { selectedMusicState } from "../../store/atoms";
import { useRecoilState } from "recoil";
import { retrieveMusicQuery } from "../../services/musics";
import {
  IconArrowBackUpDouble,
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
} from "@tabler/icons-react";

const Mp3Player = () => {
  const [selectedMusic, setSelectedMusic] = useRecoilState(selectedMusicState);
  const [mp3Url, setMp3Url] = useState("");
  const [musicId, setMusicId] = useState(0);

  useEffect(() => {
    if (selectedMusic) {
      //주석 console.log(selectedMusic.music_id);
      setMusicId(selectedMusic.music_id);
    }
  }, [selectedMusic]);

  const [musicData] = retrieveMusicQuery(musicId ? musicId : 4);
  useEffect(() => {
    if (musicData) {
      //주석 console.log(musicData);
      setMp3Url(musicData.mixed_music_source);
    }
  }, [musicData]);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const audioRef = useRef(null);

  const handleClick = () => {
    if (isPlaying) {
      // 음악이 이미 재생 중이라면 종료
      audioRef.current.pause();
      setIsPlaying(false);
      setIsDone(true);
    } else {
      // 음악 재생
      audioRef.current.play();
      setIsPlaying(true);
      setIsDone(false);
    }
  };

  useEffect(() => {
    setIsPlaying(false);
  }, [mp3Url]);

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setIsDone(true);
  };

  if (!selectedMusic) return <></>;

  return (
    <>
      {/* <button onClick={handleClick}>
        {isPlaying ? "음악 종료" : "음악 재생"}
      </button> */}
      {/* {isDone && <p>음악이 종료되었습니다.</p>} */}
      <audio ref={audioRef} src={mp3Url} onEnded={handleAudioEnded} />
      <div className="mp_playlist_player fixed h-60 w-[160] shadow-[0_0_20px_rgb(0,0,0)] transition-[0.2s] duration-[ease-in-out] rounded-[10px] top-[90%] translate-y-[-100%] left-[50%] translate-x-[-50%] bottom-5 bg-black/20 hover:cursor-pointer z-10">
        <img
          src={selectedMusic.cover_source}
          alt=""
          className="mp_playlist_player_img relative w-full h-full rounded-[10px]"
        />
        <div className="mp_playlist_controller absolute w-full h-3/6 bg-[#0000007b] backdrop-blur-[10px] flex flex-col items-center justify-center gap-[5px] rounded-[0_0_10px_10px] left-0 bottom-0">
          <div
            role="button"
            className="mp_playlist_control_img absolute bottom-2 left-5 w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-105 text-white"
            onClick={() => {
              setSelectedMusic(() => null);
            }}
          >
            <IconArrowBackUpDouble
              size={32}
              strokeWidth={2}
              color={"#ffffff"}
            />
            닫기
          </div>

          <div className="mp_playlist_control_song flex flex-col justify-center items-center ">
            <h4 className="mp_playlist_control_song_h4 text-xl text-white">
              {selectedMusic.title}
            </h4>
            <p className="mp_playlist_control_song_p text-[15px] text-[rgb(152,152,152)]">
              {selectedMusic.member.nickname}
            </p>
          </div>

          <div
            className="mp_playlist_control flex justify-center items-center gap-10"
            role="button"
            onClick={handleClick}
          >
            {isPlaying ? (
              <IconPlayerPauseFilled
                className="mp_playlist_control_img w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white"
                size={48}
                strokeWidth={2}
                color={"#ffffff"}
              />
            ) : (
              <>
                <IconPlayerPlayFilled
                  className="mp_playlist_control_img w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white"
                  size={48}
                  strokeWidth={2}
                  color={"#ffffff"}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default Mp3Player;
