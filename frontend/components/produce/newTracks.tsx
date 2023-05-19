import React, { useState } from "react";
import { useSetRecoilState } from "recoil";
import { GenreMapEntries, TagMapEntries } from "../../utils/GenreMap";
import { producingMusicState } from "../../store/atoms";
import { TagType } from "../../types/musics";
import { toastAlert } from "../../utils/toastAlert";

function NewTracks({ setTrackIds }) {
  const [genre, setGenre] = useState(null as number);
  const [tags, setTags] = useState([] as TagType[]);
  const [opacityClassName, setOpacityClassName] = useState("opacity-100");
  const setProducingMusic = useSetRecoilState(producingMusicState);

  const onGenreChange = (e) => {
    setGenre(e.target.value);
  };

  const onTagChange = (event, tag: TagType) => {
    if (event.target.checked && tags.length < 3) {
      setTags((prev) => [...prev, tag]);
    } else {
      setTags((prev) => prev.filter((e) => e !== tag));
    }
  };

  const onSubmitHandler = (e) => {
    if (genre === null) {
      return alert("장르를 선택해주세요.");
    }
    if (!tags.length) {
      return alert("태그를 선택해주세요");
    }
    //장르를 Atom에 반영하고, 투명해지면서 다음으로 넘어갑니다.
    setProducingMusic((prev) => {
      return { ...prev, genre: Number(genre), tags };
    });
    setOpacityClassName("opacity-0");
    toastAlert(`장르 선택 완료!`);
    setTimeout(() => setTrackIds([{}]), 200);
  };
  return (
    <div
      className={`Container text-gray-500 h-full w-full transition-opacity duration-200 ${opacityClassName} mt-12`}
    >
      <div className="TracksContainer mx-auto">
        <form>
          <div className="GenreDropboxContainer">
            {/* https://devdojo.com/zoltan/tailwind-css-select */}
            <label
              htmlFor="genre"
              className="libray_h3 text-2xl font-bold text-pink-500 text-[17px] mb-[15px] ml-6"
            >
              장르
            </label>
            <select
              id="genre"
              className="bg-pink-50/60 border border-pink-300 text-pink-500 text-sm rounded-lg focus:ring-pink-500 focus:border-pink-500 block w-full p-2.5 placeholder-gray-400 my-2"
              onChange={onGenreChange}
              defaultValue={"default"}
            >
              <option value="default" disabled className="hidden">
                장르를 선택하세요
              </option>
              {GenreMapEntries.map((element, index) => {
                return (
                  <option key={element[0] + index} value={index}>
                    {element[1]}
                  </option>
                );
              })}
            </select>
          </div>
          <div className="TagsCheckboxContainer pt-4">
            {/* https://www.reddit.com/r/webdev/comments/rdyyys/i_built_a_library_of_components_with_dark_mode/ */}
            {/* https://flowbite.com/docs/components/dropdowns/ */}

            <div className="libray_h3 text-2xl font-bold text-pink-500 text-[17px] ml-6">
              태그
            </div>
            <div className="mb-[15px]">태그는 3개까지 선택 가능합니다</div>
            <div className="w-full grid grid-cols-5 gap-2">
              {TagMapEntries.map((element, index) => {
                return (
                  // <div key={element[0] + index}>
                  <div
                    key={element[0] + index}
                    className="flex items-center bg-pink-50/60 backdrop-blur-[10px] border border-pink-200 rounded text-center px-4 w-full"
                  >
                    <input
                      id={element[0] + index}
                      type="checkbox"
                      value={element[0]}
                      checked={tags.includes(element[0])}
                      onChange={(event) => onTagChange(event, element[0])}
                      name="bordered-checkbox"
                      className="w-4 h-4 text-pink-600 bg-pink-100 rounded ring-2 ring-pink-300 focus:ring-pink-600  border-pink-600"
                    />
                    <label
                      htmlFor={element[0] + index}
                      className="w-full py-4 ml-2 text-sm font-medium text-pink-500"
                    >
                      {element[1]}
                    </label>
                  </div>
                  // </div>
                );
              })}
            </div>
          </div>
          <div className="flex flex-col w-full">
            <div className="mt-4 mx-auto">
              <button
                type="button"
                onClick={onSubmitHandler}
                className={`inline-block text-sm px-4 py-2 leading-none border rounded text-white border-pink-400 hover:border-transparent hover:text-pink-500 hover:bg-pink-200 mx-4 bg-pink-500 ${
                  !(genre !== null && tags.length) ? "opacity-50" : ""
                }`}
                disabled={!(genre !== null && tags.length)}
              >
                다음으로
              </button>
            </div>
            <div
              className={`mx-auto mt-2 ${
                !(genre !== null && tags.length) ? "" : "invisible"
              }`}
            >
              장르와 태그를 선택해주세요
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default NewTracks;
