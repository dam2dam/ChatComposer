import { NextApiRequest, NextApiResponse } from "next";

// 임시 데이터
const todos = [
  { id: 1, text: "Todo 1", completed: false },
  { id: 2, text: "Todo 2", completed: true },
  { id: 3, text: "Todo 3", completed: false },
];

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "GET") {
    const { id } = req.query;
    if (id) {
      const todo = todos.find((todo) => todo.id === Number(id));
      if (!todo) {
        res.status(404).end();
      } else {
        res.status(200).json(todo);
      }
    } else {
      res.status(200).json(todos);
    }
  } else if (req.method === "POST") {
    // POST 요청 처리
    const { text } = req.body;
    const newTodo = { id: todos.length + 1, text, completed: false };
    todos.push(newTodo);
    res.status(201).json(newTodo);
  } else if (req.method === "PUT") {
    // PUT 요청 처리
    const { id, completed } = req.body;
    const todoIndex = todos.findIndex((todo) => todo.id === id);
    if (todoIndex === -1) {
      res.status(404).end();
    } else {
      todos[todoIndex].completed = completed;
      res.status(200).json(todos[todoIndex]);
    }
  } else if (req.method === "DELETE") {
    // DELETE 요청 처리
    const { id } = req.query;
    const todoIndex = todos.findIndex((todo) => todo.id === Number(id));
    if (todoIndex === -1) {
      res.status(404).end();
    } else {
      const deletedTodo = todos.splice(todoIndex, 1)[0];
      res.status(200).json(deletedTodo);
    }
  } else {
    res.status(405).end();
  }
}
