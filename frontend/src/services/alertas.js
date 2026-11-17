
const tiposDeAlerta = [
  {
    campo: "prolongedEyeClosure",
    tipo: "olhos_fechados",
    mensagem: "Olhos fechados prolongadamente",
  },
  {
    campo: "yawnDetected",
    tipo: "bocejo",
    mensagem: "Bocejo detectado",
  },
  {
    campo: "excessBlinks",
    tipo: "piscadas_excessivas",
    mensagem: "Piscadas excessivas",
  },
];

export function identificarNovosAlertas(anteriores, atuais){
     return tiposDeAlerta
     .filter(({ campo }) => !anteriores[campo] && atuais[campo])
     .map(({ tipo, mensagem }) => ({ tipo, mensagem }))
}