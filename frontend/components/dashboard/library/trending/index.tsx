import React from "react";
import { SongsDummyData } from "../favorite/favoriteSong";
import TrendingSong from "./trendingSong";
import {
  listGenreMusicsQuery,
  listMusicsQuery,
} from "../../../../services/musics";
import { GenreType } from "../../../../types/musics";
import { GenreMap } from "../../../../utils/GenreMap";

function Trending({ genre = null }: { genre?: GenreType }) {
  const query = !genre ? listMusicsQuery : listGenreMusicsQuery;
  const [musics] = query(genre);
  // const genreData = GenreMapEntries.map((e) => {
  //   const [data] = listGenreMusicsQuery(e[0]);
  //   return { genreName: e[0].replaceAll("_", " "), musicList: data };
  // }).filter((e) => !!e && e.musicList.length);
  // SongsDummyData.sort((a, b) => b.favorite_count - a.favorite_count);
  // let songs: MusicListType = SongsDummyData as MusicListType;
  // const [musics] = listMusicsQuery();
  // if (data) data.sort((a, b) => b.favorite_count - a.favorite_count);
  // if (data?.length) songs = data;

  return (
    <div className="w-full mt-6">
      <h3 className="libray_h3 text-2xl font-bold text-pink-500 text-[17px] mb-[15px] ml-6">
        {genre ? `${GenreMap[genre]} 인기 순위` : "전체 인기 순위"}
      </h3>
      <div className="library_trending custom_scrollbar w-full mx-auto h-[300px] overflow-y-scroll scroll-smooth px-5 py-0 bg-pink-50/30 border-pink-500/60 border rounded-2xl">
        <table className="library_trending_table w-full text-[rgb(122, 122, 122)] border-collapse">
          <tbody>
            {musics
              ?.sort((a, b) => b.favorite_count - a.favorite_count)
              .map((song, index) => {
                return (
                  <TrendingSong
                    key={`trending_song_${song.music_id}`}
                    index={index}
                    song={song}
                  />
                );
              })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Trending;
