import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Mail, AlertCircle, CheckCircle } from "lucide-react";
import { supabase } from "@/lib/supabase";

function EmailResetPassword() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState(""); // "success" ou "error"

  const handleSendRecoveryEmail = async (e) => {
    e.preventDefault();

    if (!email) {
      setMessage("Por favor, digite seu email.");
      setMessageType("error");
      return;
    }

    setIsLoading(true);
    setMessage("");
    setMessageType("");

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });

      if (error) {
        setMessage("Erro ao enviar email. Tente novamente.");
        setMessageType("error");
      } else {
        setMessage(
          "Link de recuperação enviado com sucesso! Verifique seu email."
        );
        setMessageType("success");
        setEmail("");
      }
    } catch (err) {
      setMessage("Erro ao enviar email. Tente novamente.");
      setMessageType("error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex items-center justify-center relative overflow-hidden">
      <div className="bg-gradient-to-br from-sky-50 to-blue-50 shadow-2xl p-10 rounded-3xl w-full max-w-md relative z-10 border border-sky-100">
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <Mail className="w-10 h-10 text-sky-600" />
        </div>

        {/* Title */}
        <h1 className="text-3xl font-bold text-center text-sky-900 mb-2">
          Recuperar Senha
        </h1>

        {/* Subtitle */}
        <p className="text-center text-gray-600 mb-6 text-sm">
          Digite seu email para receber um link de recuperação de senha
        </p>

        {/* Form */}
        <form onSubmit={handleSendRecoveryEmail} className="space-y-5">
          {/* Email Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="seu@email.com"
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
            />
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

          {/* Send Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
          >
            {isLoading ? "Enviando..." : "Enviar Link de Recuperação"}
          </button>
        </form>

        {/* Back Link */}
        <div className="mt-8 pt-8 border-t border-gray-200">
          <button
            onClick={() => navigate("/")}
            className="w-full text-center text-sky-600 hover:text-sky-700 font-semibold text-sm transition"
          >
            Voltar para Login
          </button>
        </div>
      </div>
    </div>
  );
}

export default EmailResetPassword;
