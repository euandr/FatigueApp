import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./index.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

const App = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      {children}
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

// analise o projeto para ficar informado
// pode se perceber que fiz a parde de detecção. agora quero fazer a parte dos dados, onde:

// vou armazenar os dados de detecção no BD
// para gerar relatórios, entre outras coisas
// minha ideia sobre o que fazer esta um pouco vaga, preciso de sua ajuda para pensar. Além disso, preciso que me fornca um resumo de como esta o projeto, para que eu envie para outro LLM para conversar com ela.
