import React, { useEffect, useRef } from "react";
import { Dialog, Transition } from "@headlessui/react";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// @ts-ignore
const ModalWrapper: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
  const cancelButtonRef = useRef(null);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  return (
    <Transition show={isOpen} as={React.Fragment}>
      <Dialog
        as="div"
        className="fixed inset-0 z-10 overflow-y-auto"
        initialFocus={cancelButtonRef}
        onClose={onClose}
        open={isOpen}
        static
      >
        <div className="flex items-center justify-center min-h-screen">
          <Dialog.Overlay className="fixed inset-0 bg-black opacity-30" />

          <div className="bg-white rounded-md px-4 py-6 sm:p-6 md:p-8 lg:p-10 z-20">
            {children}

            <button
              className="hidden"
              onClick={onClose}
              ref={cancelButtonRef}
              type="button"
            >
              Close
            </button>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default ModalWrapper;
