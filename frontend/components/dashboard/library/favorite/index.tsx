import React, { useState } from "react";
import FavoriteSong, { SongsDummyData } from "./favoriteSong";
import { GenreType } from "../../../../types/musics";
import {
  listGenreMusicsQuery,
  listMusicsQuery,
} from "../../../../services/musics";
import { Navigation, Pagination, Scrollbar, A11y } from "swiper";
import { SwiperSlide, Swiper } from "swiper/react";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/scrollbar";
import { GenreMap } from "../../../../utils/GenreMap";
import ContentContainer from "../contentContainer";

function Favorite({ genre = null }: { genre?: GenreType }) {
  const query = !genre ? listMusicsQuery : listGenreMusicsQuery;
  const [musicsData] = query(genre);

  return (
    <div className="library_favorite w-full h-[190px] overflow-hidden mt-10 px-5 py-0">
      <h3 className="libray_h3 text-2xl font-bold text-pink-500 text-[17px] mb-[15px] ml-1">
        {genre ? `${GenreMap[genre]} 음악` : "전체 음악"}
      </h3>
      {/* <div className="library_favorite_cover flex  items-center gap-[5px] overflow-x-scroll custom_scrollbar_x"> */}
      {/* <ContentContainer> */}
      <Swiper
        modules={[Navigation, Pagination, Scrollbar, A11y]}
        autoHeight={true}
        loop={false}
        navigation={{
          prevEl: ".swiper-button-prev",
          nextEl: ".swiper-button-next",
        }}
        // pagination={{ clickable: false }}
        breakpoints={{
          1378: {
            slidesPerView: 8,
            slidesPerGroup: 8,
          },
          998: {
            slidesPerView: 6,
            slidesPerGroup: 6,
          },
          625: {
            slidesPerView: 3,
            slidesPerGroup: 3,
          },
          0: {
            slidesPerView: 1,
            slidesPerGroup: 1,
          },
        }}
      >
        {musicsData?.map((song) => {
          return (
            <SwiperSlide key={`favorite_song_${song.music_id}`}>
              <FavoriteSong song={song} />
            </SwiperSlide>
          );
        })}
        <div className="swiper-button-prev pr-8"></div>
        <div className="swiper-button-next pl-8"></div>
      </Swiper>
      {/* </ContentContainer> */}
      {/* </div> */}
    </div>
  );
}

export default Favorite;
