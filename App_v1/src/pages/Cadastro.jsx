import { useState } from "react";
import { AiOutlineEye, AiOutlineEyeInvisible } from "react-icons/ai";
import { useNavigate } from "react-router-dom";
import { MdEmail, MdLock, MdPerson } from "react-icons/md";

function Cadastro() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex items-center justify-center relative overflow-hidden">
      <div className="bg-gradient-to-br from-sky-50 to-blue-50 shadow-2xl p-10 rounded-3xl w-full max-w-md relative z-10 border border-sky-100">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-sky-500 to-blue-600 rounded-full mb-4 shadow-lg">
            <MdPerson size={30} className="text-white" />
          </div>
          <h1 className="text-4xl font-bold text-sky-900">Cadastre-se</h1>
        </div>

        <form className="space-y-5">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Nome Completo
            </label>
            <div className="relative">
              <MdPerson
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                size={20}
              />
              <input
                type="text"
                className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
                placeholder="Seu nome completo"
              />
            </div>
          </div>

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
                type={showConfirmPassword ? "text" : "password"}
                className="w-full pl-10 pr-12 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 transition bg-white"
                placeholder="••••••••"
              />

              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-sky-600 transition"
              >
                {showConfirmPassword ? (
                  <AiOutlineEyeInvisible size={20} />
                ) : (
                  <AiOutlineEye size={20} />
                )}
              </button>
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            Criar Conta
          </button>
        </form>

        <div className="mt-8 pt-8 border-t border-gray-200">
          <p className="text-center text-gray-600 text-sm">
            Já tem conta?{" "}
            <button
              onClick={() => navigate("/")}
              className="text-sky-600 cursor-pointer font-bold hover:text-sky-700 hover:underline transition"
            >
              Faça login
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Cadastro;
