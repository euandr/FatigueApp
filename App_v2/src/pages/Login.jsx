import { useState } from "react";
import {
  AiOutlineEye,
  AiOutlineEyeInvisible,
  AiOutlineLoading3Quarters,
} from "react-icons/ai";
import { useNavigate } from "react-router-dom";
import { MdEmail, MdLock } from "react-icons/md";
import { FcGoogle } from "react-icons/fc";
import { useAuthRedirect } from "@/lib/auth";
import { supabase } from "@/lib/supabase";

function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoadingGoogle, setIsLoadingGoogle] = useState(false);
  const [isLoadingForm, setIsLoadingForm] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleLogin(e) {
    e.preventDefault(); // evita recarregar a página
    setError(""); // limpa erros anteriores

    // Validação de campos vazios
    if (!email.trim()) {
      setError("Por favor, preencha o email");
      return;
    }

    if (!password.trim()) {
      setError("Por favor, preencha a senha");
      return;
    }

    setIsLoadingForm(true);

    const { data, error: authError } = await supabase.auth.signInWithPassword({
      email, // variável que vem do input de email
      password, // variável que vem do input de senha
    });

    if (authError) {
      // Mensagens de erro específicas
      if (authError.message.includes("Invalid login credentials")) {
        setError("Email ou senha incorretos");
      } else if (authError.message.includes("Email not confirmed")) {
        setError("Por favor, confirme seu email antes de fazer login");
      } else {
        setError(authError.message);
      }
      setIsLoadingForm(false);
    }
    // O redirecionamento será feito pelo useAuthRedirect quando o evento SIGNED_IN for disparado
  }

  async function handleLoginGoogle() {
    setIsLoadingGoogle(true);
    setError(""); // limpa erros anteriores
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
    });

    if (error) {
      console.log("ERRO:", error.message);
      setError("Erro ao fazer login com Google. Tente novamente.");
      setIsLoadingGoogle(false);
    }
    // O redirecionamento será feito pelo useAuthRedirect quando o evento SIGNED_IN for disparado
  }

  useAuthRedirect();

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex items-center justify-center relative overflow-hidden">
      <div className="bg-gradient-to-br from-sky-50 to-blue-50 shadow-2xl p-10 rounded-3xl w-full max-w-md relative z-10 border border-sky-100">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-sky-900">Login</h1>
        </div>

        <form className="space-y-5" onSubmit={handleLogin}>
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Email
            </label>
            <div className="relative">
              <MdEmail
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                size={20}
              />
              <input
                type="email"
                className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
                placeholder="seu@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Senha
            </label>
            <div className="relative">
              <MdLock
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                size={20}
              />
              <input
                type={showPassword ? "text" : "password"}
                className="w-full pl-10 pr-12 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />

              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-sky-600 transition"
              >
                {showPassword ? (
                  <AiOutlineEyeInvisible size={20} />
                ) : (
                  <AiOutlineEye size={20} />
                )}
              </button>
            </div>

            <button
              onClick={() => navigate("/EmailResetPassword")}
              type="button"
              className="text-sm text-sky-600 hover:text-sky-700 font-semibold mt-3 transition"
            >
              Esqueceu a senha?
            </button>
          </div>

          {error && (
            <div className="bg-red-100 border-l-4  text-red-700 p-2 rounded">
              <p className="font-semibold text-sm">{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={isLoadingForm}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isLoadingForm ? (
              <>
                <AiOutlineLoading3Quarters size={20} className="animate-spin" />
                Entrando...
              </>
            ) : (
              "Entrar"
            )}
          </button>

          <div className="relative my-10">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-gradient-to-br from-sky-50 to-blue-50 text-gray-600">
                Ou continue com
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleLoginGoogle}
            disabled={isLoadingGoogle}
            className="w-full flex items-center justify-center gap-3 bg-white border-2 border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-3 px-4 rounded-lg transition duration-300 shadow-sm hover:shadow-md disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isLoadingGoogle ? (
              <AiOutlineLoading3Quarters size={24} className="animate-spin" />
            ) : (
              <>
                <FcGoogle size={24} />
                Google
              </>
            )}
          </button>
        </form>

        <div className="mt-8 pt-8 border-t border-gray-200">
          <p className="text-center text-gray-600 text-sm">
            Não tem conta?{" "}
            <button
              onClick={() => navigate("/cadastro")}
              className="text-sky-600 cursor-pointer font-bold hover:text-sky-700 hover:underline transition"
            >
              Cadastre-se
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
export default Login;
