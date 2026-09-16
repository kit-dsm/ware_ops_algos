/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gvns_obp_1.algoritmos_routing;

import gvns_obp_1.Batch;
import gvns_obp_1.Order;
import gvns_obp_1.Product;
import gvns_obp_1.Warehouse;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 *
 * @author rasta
 */
public class S_Shape extends algoritmos_routing {

    public S_Shape(Warehouse wh) {
        super(wh);
    }

    @Override
    public float run(Batch batch) {
        float distancia;
        int max_pasillo = 0;
        int min_pasillo = Integer.MAX_VALUE;
        //float tiempo_giros = 0;
        Map Pasillos_max_dis = new HashMap();
        int num_items = 0;

        //Precalculo de distancias en hashtable
        Iterator<Order> IOrder = batch.pedidos.iterator();
        while (IOrder.hasNext()) {
            Iterator<Product> Iproductos = IOrder.next().getProductos().iterator();
            while (Iproductos.hasNext()) {
                Product p = Iproductos.next();
                num_items++;
                if (Pasillos_max_dis.containsKey(p.getPasillo())) {
                    if ((float) Pasillos_max_dis.get(p.getPasillo()) < p.getAltura()) {
                        Pasillos_max_dis.replace(p.getPasillo(), p.getAltura());
                    }
                } else {
                    Pasillos_max_dis.put(p.getPasillo(), p.getAltura());
                    if (max_pasillo < p.getPasillo()) {
                        max_pasillo = p.getPasillo();
                    }
                    if (min_pasillo > p.getPasillo()) {
                        min_pasillo = p.getPasillo();
                    }

                }
            }
        }

        //calculo distancia en los pasillos principales
        if (Pasillos_max_dis.size() % 2 == 0) {
            distancia = ((wh.getAncho_pasillos() + wh.getLargo_estanterias()) * Pasillos_max_dis.size());
        } else {
            distancia = ((wh.getAncho_pasillos() + wh.getLargo_estanterias()) * (Pasillos_max_dis.size() - 1)) + (wh.getAncho_pasillos()) + ((float) Pasillos_max_dis.get(max_pasillo) * 2);
        }
        //calculo distancia en los pasillos transversales
        if (wh.getColocacion_mesa() == 0) {
            distancia += ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (max_pasillo) * 2);
        } else if (wh.getNumero_pasillos() % 2 == 0) {
            int a = (int) (wh.getNumero_pasillos() / 2);
            distancia += ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (max_pasillo - min_pasillo) * 2);
            if (a <= min_pasillo) {
                distancia += (((wh.getAncho_pasillos() + wh.getAncho_estanterias()) / 2) + ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (min_pasillo - a))) * 2;
            } else if (a > max_pasillo) {
                distancia += (((wh.getAncho_pasillos() + wh.getAncho_estanterias()) / 2) + ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (a - max_pasillo))) * 2;
            }
        } else {
            int a = (int) (wh.getNumero_pasillos() / 2);
            distancia += ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (max_pasillo - min_pasillo) * 2);
            if (a <= min_pasillo) {
                distancia += ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (min_pasillo - a)) * 2;
            } else if (a > max_pasillo) {
                distancia += ((wh.getAncho_pasillos() + wh.getAncho_estanterias()) * (a - max_pasillo)) * 2;
            }
        }

        /*
        //calculo el numero de giros
        if (wh.getColocacion_mesa() == 0) {
            // Depot en una esquina
            if (Pasillos_max_dis.size() == 1) {
                tiempo_giros = wh.getTiempo_giro_dentro();
                if (min_pasillo != 0) {
                    tiempo_giros += (2 * wh.getTiempo_giro_fuera());
                }
            } else if (Pasillos_max_dis.size() % 2 == 0) {
                tiempo_giros = (Pasillos_max_dis.size() * 2) * wh.getTiempo_giro_fuera();
                if (min_pasillo == 0) {
                    tiempo_giros -= wh.getTiempo_giro_fuera();
                }
            } else if (Pasillos_max_dis.size() % 2 == 1) {
                tiempo_giros = wh.getTiempo_giro_dentro();
                tiempo_giros += (Pasillos_max_dis.size() * 2) * wh.getTiempo_giro_fuera();
                if (min_pasillo == 0) {
                    tiempo_giros -= wh.getTiempo_giro_fuera();
                }
            }
        } else {
            //Depot central
            if ((min_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1) && (max_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1)) {
                tiempo_giros = wh.getTiempo_giro_dentro();
            } else if (Pasillos_max_dis.size() % 2 == 0) {
                tiempo_giros = (Pasillos_max_dis.size() * 2) * wh.getTiempo_giro_fuera();
            } else if (Pasillos_max_dis.size() % 2 == 1) {
                tiempo_giros = wh.getTiempo_giro_dentro();
                tiempo_giros += (Pasillos_max_dis.size() * 2) * wh.getTiempo_giro_fuera();
                if ((min_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1) || (max_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1)) {
                    tiempo_giros -= wh.getTiempo_giro_fuera();
                }
            }
        }
         */
//        return  wh.getTiempo_depot() + (wh.getTiempo_picking() * num_items);// + tiempo_giros;
        return (distancia * wh.getVelocidad_travel()) + wh.getTiempo_depot() + (wh.getTiempo_picking() * num_items);// + tiempo_giros;

    }

}
