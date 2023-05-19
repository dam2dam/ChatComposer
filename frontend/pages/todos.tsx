import React, { useEffect } from "react";
import { useRecoilState, useRecoilValue, useSetRecoilState } from "recoil";
import { todoListState, useSsrComplectedState } from "../store/atoms";
import TodoItemCreator from "../components/TodoItemCreator";
import { removeTodo, toggleTodo } from "../store/selectors";
import Link from "next/link";

function Todos() {
  //useRecoilValue로 상태와 setRecoilValue를 가져올 수 있다.

  // useRecoilState, useRecoilValue, useSetRecoilState의 차이.
  // const todoList = useRecoilValue(todoListState);
  // const setTodoList = useSetRecoilState(todoListState);
  const [todoList, setTodoList] = useRecoilState(todoListState);
  const toggleTodoList: (todoId: number) => void =
    useSetRecoilState(toggleTodo);
  const removeTodoList: (todoId: number) => void =
    useSetRecoilState(removeTodo);

  const setSsrCompleted = useSsrComplectedState();
  useEffect(setSsrCompleted, [setSsrCompleted]);

  return (
    <div>
      <div>todos</div>
      <Link href="/queryTodos">리액트-쿼리 용으로 링크</Link>
      <TodoItemCreator />

      {todoList.map((todoItem) => (
        <div key={todoItem.id}>
          <div role="button" onClick={() => toggleTodoList(todoItem.id)}>
            {todoItem.text}
          </div>
          <div> {todoItem.completed ? "완료됨" : ""} </div>
          <div role="button" onClick={() => removeTodoList(todoItem.id)}>
            삭제하기
          </div>
        </div>
      ))}
    </div>
  );
}

export default Todos;
