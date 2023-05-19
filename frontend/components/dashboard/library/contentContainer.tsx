import React from "react";

function ContentContainer({ children }) {
  return (
    <div className="border p-2 rounded-lg bg-pink-50/30 border-pink-300">
      {children}
    </div>
  );
}

export default ContentContainer;
