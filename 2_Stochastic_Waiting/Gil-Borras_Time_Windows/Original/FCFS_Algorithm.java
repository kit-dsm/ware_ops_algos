    public List<Batch> FCFS(List<Order> Orders_iniciales) throws Exception {
        List<Batch> lotes_creados = new ArrayList<>();
        if (Orders_iniciales.size() > 0) {
            Batch b1 = new Batch(warehouse.capacidad_trabajador);

            lotes_creados.add(b1);
            List<Order> lista_pedidos_ordenados = new ArrayList<>(Orders_iniciales);
            if (lista_pedidos_ordenados.size() > 1) {
                lista_pedidos_ordenados.sort((Order o1, Order o2) -> (int) (o1.tiempo_llegada - o2.tiempo_llegada));//((o1.tiempo_llegada == o2.tiempo_llegada) ? (o2.peso - o1.peso) : (o1.tiempo_llegada - o2.tiempo_llegada)));
            }
            for (Order p1 : lista_pedidos_ordenados) {
                boolean asignado = false;

                b1 = lotes_creados.get(lotes_creados.size() - 1);
                if (b1.free_size() >= p1.peso) {
                    b1.add_order(p1);
                    asignado = true;
                }

                if (!asignado) {
                    Batch b2 = new Batch(warehouse.capacidad_trabajador);
                    try {
                        b2.add_order(p1);
                    } catch (Exception e) {
                        throw new Exception("Error en los datos: Un solo pedido supera el peso maximo del lote.");
                    }
                    lotes_creados.add(b2);

                }
            }
        }
        return lotes_creados;
    }

    public List<Batch> FCFS_compacto(List<Order> Orders_iniciales) throws Exception {
        List<Batch> lotes_creados = new ArrayList<>();
        if (Orders_iniciales.size() > 0) {
            Batch b1 = new Batch(warehouse.capacidad_trabajador);

            lotes_creados.add(b1);
            List<Order> lista_pedidos_ordenados = new ArrayList<>(Orders_iniciales);
            if (lista_pedidos_ordenados.size() > 1) {
                lista_pedidos_ordenados.sort((Order o1, Order o2) -> (int) (o1.tiempo_llegada - o2.tiempo_llegada));//((o1.tiempo_llegada == o2.tiempo_llegada) ? (o2.peso - o1.peso) : (o1.tiempo_llegada - o2.tiempo_llegada)));
            }
            for (Order p1 : lista_pedidos_ordenados) {
                boolean asignado = false;
                for (int x = 0; x < lotes_creados.size(); x++) {
                    b1 = lotes_creados.get(x);
                    if (b1.free_size() >= p1.peso) {
                        b1.add_order(p1);
                        asignado = true;
                        break;
                    }
                }

                if (!asignado) {
                    Batch b2 = new Batch(warehouse.capacidad_trabajador);
                    try {
                        b2.add_order(p1);
                    } catch (Exception e) {
                        throw new Exception("Error en los datos: Un solo pedido supera el peso maximo del lote.");
                    }
                    lotes_creados.add(b2);

                }
            }
        }
        return lotes_creados;
    }

    public List<Batch> FCFS_compacto_no_sort(List<Order> Orders_iniciales) throws Exception {
        List<Batch> lotes_creados = new ArrayList<>();
        if (Orders_iniciales.size() > 0) {
            Batch b1 = new Batch(warehouse.capacidad_trabajador);

            lotes_creados.add(b1);
            List<Order> lista_pedidos_ordenados = new ArrayList<>(Orders_iniciales);
            
            for (Order p1 : lista_pedidos_ordenados) {
                boolean asignado = false;
                for (int x = 0; x < lotes_creados.size(); x++) {
                    b1 = lotes_creados.get(x);
                    if (b1.free_size() >= p1.peso) {
                        b1.add_order(p1);
                        asignado = true;
                        break;
                    }
                }

                if (!asignado) {
                    Batch b2 = new Batch(warehouse.capacidad_trabajador);
                    try {
                        b2.add_order(p1);
                    } catch (Exception e) {
                        throw new Exception("Error en los datos: Un solo pedido supera el peso maximo del lote.");
                    }
                    lotes_creados.add(b2);

                }
            }
        }
        return lotes_creados;
    }