import { MusicDetailType } from "./../../types/musics.d";
import { useMutation, useQuery } from "@tanstack/react-query";
import TodosApi from "../TodosApi";
import { Todo } from "../../types/todos";
import { CustomQueryHookType } from "../../types/query";
import {
  useCustomMutate,
  useQueryResult,
  customInvalidate,
  prefetchingQuery,
} from "..";
import QueryKeys from "../QueryKeys";
import { GenreType, MusicListType, TagType } from "../../types/musics";
import serverApi from "../serverApi";
import { GenreMapEntries } from "../../utils/GenreMap";

export const listMusicsQuery: CustomQueryHookType<any, MusicListType> = (
  _,
  options = {},
) => {
  const queryKey = QueryKeys.musics.list.all();
  const queryFn = () => serverApi.get("/musics");

  return useQueryResult(queryKey, queryFn, options);
};

export const listGenreMusicsQuery: CustomQueryHookType<
  GenreType,
  MusicListType
> = (genre, options = {}) => {
  const queryKey = QueryKeys.musics.list.genre(genre);
  const genreIndex = GenreMapEntries.findIndex((e) => e[0] === genre);
  const queryFn = () => serverApi.get(`/musics?genre=${genreIndex}`);

  return useQueryResult(queryKey, queryFn, options);
};

export const listTagMusicsQuery: CustomQueryHookType<TagType, MusicListType> = (
  tag,
  options = {},
) => {
  const queryKey = QueryKeys.musics.list.tag(tag);
  const queryFn = () => serverApi.get(`/musics?tag=${tag}`);

  return useQueryResult(queryKey, queryFn, options);
};

export const retrieveMusicQuery: CustomQueryHookType<
  number,
  MusicDetailType
> = (musicId: number, options = {}) => {
  const queryKey = QueryKeys.musics.retrieve(musicId);
  const queryFn = () => serverApi.get(`/musics/${musicId}`);
  return useQueryResult(queryKey, queryFn, options);
};

export const prefetchListMusicsQuery = () => {
  const queryKey = QueryKeys.musics.list.all();
  const queryFn = () => serverApi.get("/musics");
  return prefetchingQuery(queryKey, queryFn);
};

export const prefetchListGenreMusicsQuery = (genre: GenreType) => {
  const queryKey = QueryKeys.musics.list.genre(genre);
  const genreIndex = GenreMapEntries.findIndex((e) => e[0] === genre);
  const queryFn = () => serverApi.get(`/musics?genre=${genreIndex}`);
  return prefetchingQuery(queryKey, queryFn);
};

export const prefetchListTagMusicsQuery = (tag: TagType) => {
  const queryKey = QueryKeys.musics.list.tag(tag);
  const queryFn = () => serverApi.get(`/musics?tag=${tag}`);
  return prefetchingQuery(queryKey, queryFn);
};

export const listTodoQuery: CustomQueryHookType<null, Array<Todo>> = (
  _,
  options = {},
) => {
  const queryKey = QueryKeys.todos.list();
  const queryFn = () => TodosApi.get("");

  return useQueryResult(queryKey, queryFn, options);
};

export const retrieveTodoQuery: CustomQueryHookType<number, Todo> = (
  selectedId,
  options = {},
) => {
  const queryKey = QueryKeys.todos.retrieve(selectedId);
  const queryFn = () => TodosApi.get(`?id=${selectedId}`);

  return useQueryResult(queryKey, queryFn, options);
};

export const useCreateTodoMutate = () => {
  const mutationFn = (text: string) => TodosApi.post("", { text });
  const queryKey = QueryKeys.todos.list();

  return useCustomMutate(queryKey, mutationFn);
};

export const useUpdateTodoMutate = () => {
  const mutationFn = (todo: Todo) => TodosApi.put("", { ...todo });
  const queryKey = QueryKeys.todos.list();

  return useCustomMutate(queryKey, mutationFn);
};

export const useDestoryTodoMutate = () => {
  const mutationFn = (id: number) => TodosApi.delete(`?id=${id}`);
  const queryKey = QueryKeys.todos.list();

  return useCustomMutate(queryKey, mutationFn);
};

const todosQuery = {
  listTodoQuery,
  retrieveTodoQuery,
  useCreateTodoMutate,
  useUpdateTodoMutate,
  useDestoryTodoMutate,
};

export default todosQuery;
