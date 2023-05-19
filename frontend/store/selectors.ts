import { selector } from "recoil";
import { todoListState } from "./atoms";
import { Todo } from "../types/todos";

export const todoListSelector = selector<Todo[]>({
  key: "todoListSelector",
  get: ({ get }) => get(todoListState),
  set: ({ set }, newValue) => set(todoListState, newValue),
});

export const addTodo = selector({
  key: "addTodo",
  get: ({ get }) => get(todoListSelector),
  set: ({ get, set }, newTodo) => {
    const todoList = get(todoListSelector);
    const updatedTodoList = [...todoList, newTodo];
    set(todoListSelector, updatedTodoList);
  },
});

export const removeTodo = selector({
  key: "removeTodo",
  get: ({ get }) => undefined,
  set: ({ get, set }, todoId) => {
    const todoList = get(todoListSelector);
    const updatedTodoList = todoList.filter((todo) => todo.id !== todoId);
    set(todoListSelector, updatedTodoList);
  },
});

export const toggleTodo = selector({
  key: "toggleTodo",
  get: ({ get }) => undefined,
  set: ({ get, set }, todoId: number) => {
    const todoList = get(todoListSelector);
    const updatedTodoList = todoList.map((todo) =>
      todo.id === todoId ? { ...todo, completed: !todo.completed } : todo,
    );
    set(todoListSelector, updatedTodoList);
  },
});
