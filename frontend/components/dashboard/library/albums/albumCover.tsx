import React from "react";
import { AlbumCoverType } from "../../../../types/musics";
import {
  IconListNumbers,
  IconPlaylist,
  IconPlaylistAdd,
} from "@tabler/icons-react";

function AlbumCover({ genreName = "ALL", coverImgSrc, numOfSongs }) {
  const onAddPlaylist = () => {};
  return (
    <div
      className="library_album_covers relative w-[190px] h-[190px] border border-pink-300 border-collapse flex flex-col mx-auto flex-shrink-0 justify-end transition-[0.2s] duration-[ease-in-out] p-2.5 rounded-[10px] hover:cursor-pointer"
      onClick={onAddPlaylist}
    >
      <div className="bg-black absolute rounded top-0 left-0 w-[190px] h-[190px] z-[6] opacity-0 hover:opacity-60">
        <IconListNumbers
          className="library_album_covers_img absolute z-[4] w-[100px] top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%] 가운데에넣기"
          size={48}
          strokeWidth={2}
          color={"#ffffff"}
        />
      </div>
      <img
        src={coverImgSrc}
        alt=""
        className="album_cover absolute w-[98%] rounded-[25px] left-0"
      />
      {/* <img
        className="library_album_covers_img absolute z-[4] w-[30px] right-2.5 bottom-2.5 bg-black"
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAApElEQVRIie2TPQ7CMAxGHcTIlTgBVD0UQuqJuhVFQt04CSNlf1060BTF+etQqU/KZn8vih2RnZIAPX6ebo+JFKDVGGNmmYcYQQq7wGUx5GNKijtIH5t7ojABUAMW+AJvoAXORYxAo/xW9bP5wmstPFdgSwtm6wYMInJSm1Ze009MsSt4BfSE1PwHqAJGcEkWTJK7J/yWFf4juQIPYJhOl33zzTIC7lwSFRt0S08AAAAASUVORK5CYII="
      /> */}
      {/* <IconPlaylistAdd
        className="library_album_covers_img absolute z-[4] w-[30px] right-2.5 bottom-2.5"
        size={48}
        strokeWidth={2}
        color={"#ffffff"}
      /> */}
      {/* <img
        className="library_album_covers_img absolute z-[4] w-[30px] right-2.5 bottom-2.5 bg-black"
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAApElEQVRIie2TPQ7CMAxGHcTIlTgBVD0UQuqJuhVFQt04CSNlf1060BTF+etQqU/KZn8vih2RnZIAPX6ebo+JFKDVGGNmmYcYQQq7wGUx5GNKijtIH5t7ojABUAMW+AJvoAXORYxAo/xW9bP5wmstPFdgSwtm6wYMInJSm1Ze009MsSt4BfSE1PwHqAJGcEkWTJK7J/yWFf4juQIPYJhOl33zzTIC7lwSFRt0S08AAAAASUVORK5CYII="
      /> */}
      <h5 className="library_album_covers_h5 text-white text-[15px] z-[2]">
        {genreName}
      </h5>
      <p className="library_album_covers_p text-white text-xs z-[2]">
        {numOfSongs}개의 음악이 있습니다.
      </p>
      <span className="library_album_covers_span absolute w-full h-[60px] bg-[#000000b7] backdrop-blur-[10px] rounded-[0_0_10px_10px] left-0 bottom-0" />
    </div>
  );
}

export default AlbumCover;

export const DummyCoversData: AlbumCoverType[] = [
  {
    member: { member_id: 1, nickname: "아무닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 3,
    genre: "Hip-Hop",
    music_id: 1,
    tags: ["사랑", "기쁨", "슬픔"],
    title: "10 new songs",
    is_my_favorite: "n",
  },
  {
    member: { member_id: 2, nickname: "건전한닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1505672984986-b7c468c7a134?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 7,
    genre: "Electronic",
    music_id: 2,
    tags: ["즐거움"],
    title: "20 new songs",
    is_my_favorite: "y",
  },
  {
    member: { member_id: 1, nickname: "아무닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1614613535308-eb5fbd3d2c17?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 2,
    genre: "Pop",
    music_id: 3,
    tags: ["개쩌는", "신나는"],
    title: "30 new songs",
    is_my_favorite: "y",
  },
  {
    member: { member_id: 3, nickname: "스티브잡스" },
    cover_source:
      "https://images.unsplash.com/photo-1571330735066-03aaa9429d89?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 0,
    genre: "Rock",
    music_id: 4,
    tags: ["사랑", "슬픔"],
    title: "40 new songs",
    is_my_favorite: "n",
  },
  {
    member: { member_id: 3, nickname: "스티브잡스" },
    cover_source:
      "https://images.unsplash.com/photo-1571330735066-03aaa9429d89?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 2325,
    genre: "R&B",
    music_id: 5,
    tags: ["대충아무감정"],
    title: "50 new songs",
    is_my_favorite: "y",
  },
  {
    member: { member_id: 1, nickname: "아무닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 3,
    genre: "Hip-Hop",
    music_id: 6,
    tags: ["사랑", "기쁨", "슬픔"],
    title: "10 new songs",
    is_my_favorite: "n",
  },
  {
    member: { member_id: 2, nickname: "건전한닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1505672984986-b7c468c7a134?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 7,
    genre: "Electronic",
    music_id: 7,
    tags: ["즐거움"],
    title: "20 new songs",
    is_my_favorite: "n",
  },
  {
    member: { member_id: 1, nickname: "아무닉네임" },
    cover_source:
      "https://images.unsplash.com/photo-1614613535308-eb5fbd3d2c17?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 2,
    genre: "Pop",
    music_id: 8,
    tags: ["개쩌는", "신나는"],
    title: "30 new songs",
    is_my_favorite: "n",
  },
  {
    member: { member_id: 3, nickname: "스티브잡스" },
    cover_source:
      "https://images.unsplash.com/photo-1571330735066-03aaa9429d89?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 0,
    genre: "Rock",
    music_id: 9,
    tags: ["사랑", "슬픔"],
    title: "40 new songs",
    is_my_favorite: "y",
  },
  {
    member: { member_id: 3, nickname: "스티브잡스" },
    cover_source:
      "https://images.unsplash.com/photo-1571330735066-03aaa9429d89?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80",
    created_at: String(Date.now()),
    favorite_count: 2325,
    genre: "R&B",
    music_id: 10,
    tags: ["대충아무감정"],
    title: "50 new songs",
    is_my_favorite: "y",
  },
];
