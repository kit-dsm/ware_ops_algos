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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 *
 * @author rasta
 */
public class Largest_Gap extends algoritmos_routing {

    public Largest_Gap(Warehouse wh) {
        super(wh);
    }

    @Override
    public float run(Batch batch) {
        float distancia = 0;
        int max_pasillo = 0;
        int min_pasillo = Integer.MAX_VALUE;
        float tiempo_giros;
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
                    List<Float> lista_alturas = (List<Float>) Pasillos_max_dis.get(p.getPasillo());
                    if (!lista_alturas.contains(p.getAltura())) {
                        lista_alturas.add(p.getAltura());
                        Pasillos_max_dis.replace(p.getPasillo(), lista_alturas);
                    }
                } else {
                    List<Float> lista_alturas;
                    lista_alturas = new ArrayList<>();
                    lista_alturas.add(p.getAltura());
                    lista_alturas.add((float) 0);
                    lista_alturas.add((float) wh.getLargo_estanterias());
                    //lista_alturas.add(p.getAltura());
                    Pasillos_max_dis.put(p.getPasillo(), lista_alturas);
                    if (max_pasillo < p.getPasillo()) {
                        max_pasillo = p.getPasillo();
                    }
                    if (min_pasillo > p.getPasillo()) {
                        min_pasillo = p.getPasillo();
                    }

                }
            }
        }

        Map Pasillos_gap_locate = new HashMap();
        //0=gap en el inicio del pasillo
        //1=gap en medio del pasillo
        //2=gap al final del pasillo

        //calculo distancias en pasillos principales
        if (Pasillos_max_dis.size() > 2) {
            distancia = ((wh.getAncho_pasillos() + wh.getLargo_estanterias()) * 2);
            Iterator entries = Pasillos_max_dis.entrySet().iterator();
            while (entries.hasNext()) {
                Map.Entry thisEntry = (Map.Entry) entries.next();
                if (!thisEntry.getKey().equals(min_pasillo) && !thisEntry.getKey().equals(max_pasillo)) {
                    List<Float> lista_alturas = (List<Float>) thisEntry.getValue();
                    Collections.sort(lista_alturas);
                    float max_gap = 0;
                    for (int i = 1; i < lista_alturas.size(); i++) {
                        float dd = (float) lista_alturas.get(i) - (float) lista_alturas.get(i - 1);
                        if (max_gap < dd) {
                            if (i == 1) {
                                Pasillos_gap_locate.put(thisEntry.getKey(), 0);
                            } else if (i == lista_alturas.size() - 1) {
                                Pasillos_gap_locate.put(thisEntry.getKey(), 2);
                            } else {
                                Pasillos_gap_locate.put(thisEntry.getKey(), 1);
                            }
                            max_gap = dd;
                        }
                    }
                    distancia += wh.getAncho_pasillos() + ((wh.getLargo_estanterias() - max_gap) * 2);
                    if ((int) Pasillos_gap_locate.get(thisEntry.getKey()) == 1) {
                        distancia += wh.getAncho_pasillos();
                    }
                }

            }
        } else if (Pasillos_max_dis.size() > 1) {
            distancia = ((wh.getAncho_pasillos() + wh.getLargo_estanterias()) * 2);
        } else {
            Iterator entries = Pasillos_max_dis.entrySet().iterator();
            while (entries.hasNext()) {
                Map.Entry thisEntry = (Map.Entry) entries.next();
                List<Float> lista_alturas = (List<Float>) thisEntry.getValue();
                Collections.sort(lista_alturas);
                distancia = wh.getAncho_pasillos() + ((lista_alturas.get(lista_alturas.size() - 2)) * 2);
            }
        }

        //calculo de distancia en pasillos transversales
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
            } else {
                tiempo_giros = (Pasillos_max_dis.size() * 2 * wh.getTiempo_giro_fuera());
                tiempo_giros += (Pasillos_gap_locate.size() * 2 * wh.getTiempo_giro_dentro());
                if (min_pasillo == 0) {
                    tiempo_giros -= wh.getTiempo_giro_fuera();
                }
                int num_gap_medio = 0;
                Iterator entries = Pasillos_gap_locate.entrySet().iterator();
                while (entries.hasNext()) {
                    Map.Entry thisEntry = (Map.Entry) entries.next();
                    if ((int) thisEntry.getValue() == 1) {
                        num_gap_medio++;
                    }
                }
                tiempo_giros += (num_gap_medio * 2 * wh.getTiempo_giro_fuera());
                tiempo_giros += (num_gap_medio * wh.getTiempo_giro_dentro());
            }
        } else {
            //Depot central
            if ((min_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1) && (max_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1)) {
                tiempo_giros = wh.getTiempo_giro_dentro();
            } else {
                tiempo_giros = (Pasillos_max_dis.size() * 2 * wh.getTiempo_giro_fuera());
                tiempo_giros += (Pasillos_gap_locate.size() * 2 * wh.getTiempo_giro_dentro());
                if ((min_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1) || (max_pasillo == (int) (Pasillos_max_dis.size() / 2) + 1)) {
                    tiempo_giros -= wh.getTiempo_giro_fuera();
                }
                int num_gap_medio = 0;
                Iterator entries = Pasillos_gap_locate.entrySet().iterator();
                while (entries.hasNext()) {
                    Map.Entry thisEntry = (Map.Entry) entries.next();
                    if ((int) thisEntry.getValue() == 1) {
                        num_gap_medio++;
                    }
                }
                tiempo_giros += (num_gap_medio * 2 * wh.getTiempo_giro_fuera());
                tiempo_giros += (num_gap_medio * wh.getTiempo_giro_dentro());

            }
        }
         */
        return (distancia * wh.getVelocidad_travel()) + wh.getTiempo_depot() + (wh.getTiempo_picking() * num_items);// + tiempo_giros;

    }

}
