import React, { useEffect, useState } from "react";
import { AlbumCoverType, GenreType, MusicType } from "../../../../types/musics";
import { IconHeart, IconHeartFilled } from "@tabler/icons-react";
import serverApi from "../../../../services/serverApi";
import { toastAlert } from "../../../../utils/toastAlert";
import { customInvalidate } from "../../../../services";
import QueryKeys from "../../../../services/QueryKeys";
import { TagMap } from "../../../../utils/GenreMap";

function TrendingSong({ song, index }: { song: MusicType; index: number }) {
  const {
    music_id: musicId,
    cover_source: coverSource,
    member: { nickname },
    title,
    is_my_favorite,
    favorite_count,
    tags,
    genre,
  } = song;

  const [isLiked, setIsLiked] = useState(is_my_favorite === "y");

  useEffect(() => {
    //주석 console.log(is_my_favorite);
    setIsLiked(is_my_favorite === "y");
  }, [is_my_favorite]);
  //좋아요 표시

  return (
    <tr className="library_trending_table_tr transition-[0.2s] duration-[ease-in-out] border-t-[none] border-solid border-y-[0.2px] border-x-0 border-[rgba(125,125,125,0.419)] hover:cursor-pointer hover:bg-[rgba(0, 0, 0, 0.2)]">
      <td className="library_trending_table_tr_td p-2.5">
        <p className="library_trending_table_tr_td_p text-[13px]">
          {index + 1} 위
        </p>
      </td>
      <td>
        <img
          src={coverSource}
          alt=""
          className="song_cover w-[60px] rounded-lg"
        />
      </td>
      <td className="song p-2.5 flex flex-col gap-[5px]">
        <h4 className="song_h4 text-[15px] text-pink-500">{title}</h4>
        <p className="song_p text-[13px] text-[rgb(184, 184, 184)]">
          {nickname}
        </p>
      </td>
      <td className="library_trending_table_tr_td p-2.5">
        <p className="library_trending_table_tr_td_p text-[13px]">{genre}</p>
      </td>
      <td className="library_trending_table_tr_td p-2.5">
        <p className="library_trending_table_tr_td_p text-[13px]">
          좋아요 {favorite_count} 개
        </p>
      </td>
      <td className="library_trending_table_tr_td p-2.5">
        <p className="library_trending_table_tr_td_p text-[13px]">
          {tags.length ? "#" + tags.map((e) => TagMap[e]).join(" #") : ""}
        </p>
      </td>
      <td className="library_trending_table_tr_td p-2.5">
        {/* <img className="library_trending_img w-5" src={likedImage} /> */}
        {/* <img className="library_trending_img w-5" src={likedImage} /> */}
        {isLiked ? (
          <IconHeartFilled
            className="text-pink-500"
            onClick={() => {
              serverApi
                .delete(`/musics/${musicId}`)
                .then((res) => {
                  toastAlert("좋아요를 취소하였습니다.");
                  customInvalidate(QueryKeys.musics.list.all());
                  customInvalidate(
                    QueryKeys.musics.list.genre(genre as GenreType),
                  );
                  setIsLiked(false);
                })
                .catch((err) => setIsLiked((prev) => !prev));
            }}
          />
        ) : (
          <IconHeart
            className="text-pink-500"
            onClick={() => {
              serverApi
                .post(`/musics/${musicId}`)
                .then((res) => {
                  toastAlert("좋아요를 등록하였습니다.");
                  customInvalidate(QueryKeys.musics.list.all());
                  customInvalidate(
                    QueryKeys.musics.list.genre(genre as GenreType),
                  );
                  setIsLiked(true);
                })
                .catch((err) => setIsLiked((prev) => !prev));
            }}
          />
        )}
      </td>
    </tr>
  );
}

export default TrendingSong;
