import React, { useEffect, useState } from "react";
import AlbumCover, { DummyCoversData } from "./albumCover";
import { GenreMapEntries } from "../../../../utils/GenreMap";
import {
  listGenreMusicsQuery,
  listMusicsQuery,
} from "../../../../services/musics";
import { MusicType } from "../../../../types/musics";
import { Navigation, Pagination, Scrollbar, A11y } from "swiper";
import { SwiperSlide, Swiper } from "swiper/react";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/scrollbar";
import { useRecoilState, useSetRecoilState } from "recoil";
import { genreDataState, selectedGenreState } from "../../../../store/atoms";
import ContentContainer from "../contentContainer";

function Albums() {
  let genreData = GenreMapEntries.map((e) => {
    const [data] = listGenreMusicsQuery(e[0]);
    //주석 console.log(data);
    return { genreName: e[0], musicList: data };
  }).filter((e) => !!e && e.musicList?.length);

  const [snapshot, setGenreDataState] = useRecoilState(genreDataState);
  useEffect(() => {
    if (genreData.length) {
      const genreSet = genreData.map((e) => {
        return e.genreName;
      });
      if (JSON.stringify(genreSet) !== JSON.stringify(snapshot)) {
        setGenreDataState(genreSet);
      }
    }
  }, [genreData]);

  const [allMusicData] = listMusicsQuery();
  const setSelectedGenre = useSetRecoilState(selectedGenreState);
  if (!genreData || !genreData.length) return <></>;

  return (
    <div className="library_album mt-8">
      <h3 className="libray_h3 text-2xl font-bold text-pink-500 text-[17px] mb-[15px] ml-6">
        {/* Recently Listened Albums */}
        장르별 음악
      </h3>
      <ContentContainer>
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
              slidesPerView: 6,
              slidesPerGroup: 6,
            },
            998: {
              slidesPerView: 5,
              slidesPerGroup: 5,
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
          <SwiperSlide>
            <div className="" onClick={() => setSelectedGenre(null)}>
              <AlbumCover
                coverImgSrc={allMusicData?.at(-1).cover_source}
                numOfSongs={allMusicData?.length}
              />
            </div>
          </SwiperSlide>
          {allMusicData ? (
            genreData.map(({ genreName, musicList }, index) => {
              return (
                <SwiperSlide key={`${genreName}_${index}`}>
                  <div className="" onClick={() => setSelectedGenre(genreName)}>
                    <AlbumCover
                      genreName={genreName.replaceAll("_", " ")}
                      coverImgSrc={musicList?.at(-1).cover_source}
                      numOfSongs={musicList?.length}
                    />
                  </div>
                </SwiperSlide>
              );
            })
          ) : (
            <></>
          )}

          <div className="swiper-button-prev pr-8"></div>
          <div className="swiper-button-next pl-8"></div>
        </Swiper>
      </ContentContainer>
    </div>
  );
}

export default Albums;
