import React, { useState } from "react";
import { useSetRecoilState } from "recoil";
import { todoListState } from "../store/atoms";

function TodoItemCreator() {
  const [inputValue, setInputValue] = useState("");
  //useSetRecoilState로 상태를 수정할 수 있다.
  const setTodoList = useSetRecoilState(todoListState);
  const addItem = () => {
    // @ts-ignore
    setTodoList((prev) => [
      ...prev,
      { id: getId(), text: inputValue, isComplete: false },
    ]);
    setInputValue("");
  };

  const onChange = ({ target: { value } }) => {
    setInputValue(value);
  };

  return (
    <div>
      <div>todoItemCreator</div>
      <input type="text" value={inputValue} onChange={onChange} />
      <button onClick={addItem}>추가하기</button>
    </div>
  );
}

export default TodoItemCreator;

let id = 0;
function getId() {
  return id++;
}
