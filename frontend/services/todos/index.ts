import { useMutation, useQuery } from "@tanstack/react-query";
import TodosApi from "../TodosApi";
import { Todo } from "../../types/todos";
import { CustomQueryHookType } from "../../types/query";
import { useCustomMutate, useQueryResult, customInvalidate } from "..";
import QueryKeys from "../QueryKeys";

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
