import { supabase } from "./supabase";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export async function handleCadastro({ nome, email, senha, confirmSenha }) {
  // Validar campos
  if (!nome || !email || !senha || !confirmSenha) {
    return { error: "Preencha todos os campos" };
  }

  if (senha !== confirmSenha) {
    return { error: "Senhas não conferem" };
  }

  if (!/^(?=.*[A-Z])(?=.*\d).{6,}$/.test(senha)) {
    return {
      error: "Senha precisa ter 8 caracteres, 1 número e 1 letra maiúscula",
    };
  }

  try {
    const { data, error } = await supabase.auth.signUp({
      email,
      password: senha,
      options: { data: { nome } },
    });

    if (error) return { error: error.message };
    return { data };
  } catch (err) {
    return { error: err.message };
  }
}

export function useAuthRedirect() {
  const navigate = useNavigate();

  useEffect(() => {
    // Escutar mudanças de autenticação
    const { data: authListener } = supabase.auth.onAuthStateChange(
      (event, session) => {
        // Redirecionar apenas quando o evento é SIGNED_IN
        if (event === "SIGNED_IN") {
          // Pequeno delay para garantir que a sessão está completamente configurada
          setTimeout(() => {
            navigate("/monitoramento");
          }, 100);
        }
      }
    );

    return () => {
      authListener?.subscription?.unsubscribe();
    };
  }, [navigate]);
}
