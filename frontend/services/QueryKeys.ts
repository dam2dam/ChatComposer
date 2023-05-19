import { GenreType, TagType } from "../types/musics";
import { GenreMapEntries } from "../utils/GenreMap";

const QueryKeys = {
  todos: {
    list: () => ["todo-list"] as const,
    retrieve: (TodoId: number) => [...QueryKeys.todos.list(), TodoId] as const,
  },
  musics: {
    list: {
      all: () => ["musics"] as const,
      genre: (genre: GenreType) =>
        [...QueryKeys.musics.list.all(), "genre", genre] as const,
      tag: (tag: TagType) =>
        [...QueryKeys.musics.list.all(), "tag", tag] as const,
    },
    retrieve: (musicId: number) =>
      [...QueryKeys.musics.list.all(), musicId] as const,
  },
};

export default QueryKeys;
