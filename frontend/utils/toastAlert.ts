import { toast } from "react-toastify";

export const toastAlert = (message: string) =>
  toast.success(message, {
    position: "top-center",
    autoClose: 500,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
  });
