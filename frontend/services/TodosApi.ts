import axios from "axios";

const TodosApi = axios.create({
  baseURL: "http://localhost:3000/api/todos",
});

export default TodosApi;
