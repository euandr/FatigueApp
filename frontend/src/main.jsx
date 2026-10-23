import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import Login from "./pages/Login.jsx";
import Cadastro from "./pages/Cadastro.jsx";
import Monitoramento from "./pages/Monitoramento.jsx";
import NotFound from "./pages/NotFound.jsx";
import ResetPassword from "./pages/ResetPassword.jsx";
import EmailResetPassword from "./pages/EmailResetPassword.jsx";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <App>
        <Login />
      </App>
    ),
  },
  {
    path: "/cadastro",
    element: (
      <App>
        <Cadastro />
      </App>
    ),
  },
  {
    path: "/monitoramento",
    element: (
      <App>
        <Monitoramento />
      </App>
    ),
  },
  {
    path: "/EmailResetPassword",
    element: (
      <App>
        <EmailResetPassword />
      </App>
    ),
  },
  {
    path: "/reset-password",
    element: (
      <App>
        <ResetPassword />
      </App>
    ),
  },
  {
    path: "*",
    element: (
      <App>
        <NotFound />
      </App>
    ),
  },
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
