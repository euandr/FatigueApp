import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AiOutlineEye, AiOutlineEyeInvisible } from "react-icons/ai";
import { MdLock } from "react-icons/md";
import { AlertCircle, CheckCircle } from "lucide-react";
import { supabase } from "@/lib/supabase";

function ResetPassword() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [showPasswordConfirm, setShowPasswordConfirm] = useState(false);
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("");
  const [isValidLink, setIsValidLink] = useState(true);

  useEffect(() => {
    // Verifica se o usuário foi redirecionado do email de reset
    const hash = window.location.hash;
    if (!hash.includes("access_token")) {
      setMessage("Link de recuperação inválido ou expirado.");
      setMessageType("error");
      setIsValidLink(false);
    }
  }, []);

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setMessage("");
    setMessageType("");

    // Validações
    if (!password || !passwordConfirm) {
      setMessage("Por favor, preencha todos os campos.");
      setMessageType("error");
      return;
    }

    if (password.length < 6) {
      setMessage("A senha deve ter no mínimo 6 caracteres.");
      setMessageType("error");
      return;
    }

    if (password !== passwordConfirm) {
      setMessage("As senhas não coincidem.");
      setMessageType("error");
      return;
    }

    setIsLoading(true);

    try {
      const { error } = await supabase.auth.updateUser({
        password: password,
      });

      if (error) {
        setMessage(error.message || "Erro ao atualizar a senha.");
        setMessageType("error");
      } else {
        setMessage("Senha atualizada com sucesso! Redirecionando...");
        setMessageType("success");
        setTimeout(() => {
          navigate("/");
        }, 2000);
      }
    } catch (err) {
      setMessage("Erro ao atualizar a senha. Tente novamente.");
      setMessageType("error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex items-center justify-center relative overflow-hidden">
      <div className="bg-gradient-to-br from-sky-50 to-blue-50 shadow-2xl p-10 rounded-3xl w-full max-w-md relative z-10 border border-sky-100">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-sky-900">Redefinir Senha</h1>
          <p className="text-gray-600 text-sm mt-2">Digite sua nova senha</p>
        </div>

        {!isValidLink ? (
          <div className="space-y-5">
            {/* Feedback Message - Link Inválido */}
            {message && (
              <div
                className={`flex items-center gap-2 p-4 rounded-lg text-sm ${
                  messageType === "success"
                    ? "bg-sky-50 text-sky-700"
                    : "bg-red-50 text-red-700"
                }`}
              >
                {messageType === "success" ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
                {message}
              </div>
            )}

            {/* Botão Voltar */}
            <button
              onClick={() => navigate("/")}
              className="w-full bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Voltar para Login
            </button>
          </div>
        ) : (
          <>
            <form className="space-y-5" onSubmit={handleResetPassword}>
              {/* Nova Senha */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Nova Senha
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
              </div>

              {/* Confirmar Senha */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Confirmar Senha
                </label>
                <div className="relative">
                  <MdLock
                    className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    size={20}
                  />
                  <input
                    type={showPasswordConfirm ? "text" : "password"}
                    className="w-full pl-10 pr-12 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
                    placeholder="••••••••"
                    value={passwordConfirm}
                    onChange={(e) => setPasswordConfirm(e.target.value)}
                  />

                  <button
                    type="button"
                    onClick={() => setShowPasswordConfirm(!showPasswordConfirm)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-sky-600 transition"
                  >
                    {showPasswordConfirm ? (
                      <AiOutlineEyeInvisible size={20} />
                    ) : (
                      <AiOutlineEye size={20} />
                    )}
                  </button>
                </div>
              </div>

              {/* Feedback Message */}
              {message && (
                <div
                  className={`flex items-center gap-2 p-3 rounded-lg text-sm ${
                    messageType === "success"
                      ? "bg-sky-50 text-sky-700"
                      : "bg-red-50 text-red-700"
                  }`}
                >
                  {messageType === "success" ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <AlertCircle className="w-5 h-5" />
                  )}
                  {message}
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
              >
                {isLoading ? "Atualizando..." : "Atualizar Senha"}
              </button>
            </form>

            <div className="mt-8 pt-8 border-t border-gray-200">
              <button
                onClick={() => navigate("/")}
                className="w-full text-center text-sky-600 hover:text-sky-700 font-semibold text-sm transition"
              >
                Voltar para Login
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default ResetPassword;
