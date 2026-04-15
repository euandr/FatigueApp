import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/supabase";
import { createOrGetDevice, getActiveDevices } from "@/lib/queries/devices";

/**
 * Hook para gerenciar devices do usuário
 * Registra device quando câmera abre, carrega lista de devices ativos
 */
export function useDevices() {
  const [currentDevice, setCurrentDevice] = useState(null);
  const [devices, setDevices] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Registra ou obtém device quando câmera é iniciada
  const registerDeviceOnCamera = useCallback(async (deviceName = null) => {
    try {
      setIsLoading(true);
      setError(null);

      // Pega usuário autenticado
      const {
        data: { user },
        error: authError,
      } = await supabase.auth.getUser();

      if (authError || !user) {
        throw new Error("Usuário não autenticado");
      }

      // Cria ou obtém device
      const { data, error: deviceError } = await createOrGetDevice(
        user.id,
        deviceName,
      );

      if (deviceError) throw deviceError;

      setCurrentDevice(data);
      return { success: true, device: data };
    } catch (err) {
      const errorMsg = err.message || "Erro ao registrar device";
      setError(errorMsg);
      console.error("Device registration error:", err);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Carrega lista de devices ativos
  const loadActiveDevices = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const {
        data: { user },
        error: authError,
      } = await supabase.auth.getUser();

      if (authError || !user) {
        throw new Error("Usuário não autenticado");
      }

      const { data, error: devicesError } = await getActiveDevices(user.id);

      if (devicesError) throw devicesError;

      setDevices(data || []);
      return { success: true, devices: data };
    } catch (err) {
      const errorMsg = err.message || "Erro ao carregar devices";
      setError(errorMsg);
      console.error("Load devices error:", err);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Carrega devices ao montar componente
  useEffect(() => {
    loadActiveDevices();
  }, [loadActiveDevices]);

  // Sincroniza currentDevice quando devices carrega
  useEffect(() => {
    if (devices.length > 0 && !currentDevice) {
      console.log(
        "✅ Sincronizando currentDevice com device antigo:",
        devices[0].id,
      );
      setCurrentDevice(devices[0]);
    }
  }, [devices, currentDevice]);

  return {
    currentDevice,
    devices,
    isLoading,
    error,
    registerDeviceOnCamera,
    loadActiveDevices,
    setCurrentDevice,
  };
}
