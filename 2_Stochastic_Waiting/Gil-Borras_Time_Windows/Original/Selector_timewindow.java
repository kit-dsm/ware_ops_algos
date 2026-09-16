private algoritmos_timeWindow selector_alg_timeWindow_string(String alg_timeWindow, Warehouse w, algoritmos alg_routing) {
        double threshold = 0.3;
        switch (alg_timeWindow) {
            case "henn_0.0"://{0,0.25,0.5,0.75,1}
                return new timeWindow_henn(w, alg_routing, 0.0);
            case "henn_0.25"://{0,0.25,0.5,0.75,1}
                return new timeWindow_henn(w, alg_routing, 0.25);
            case "henn_0.5"://{0,0.25,0.5,0.75,1}
                return new timeWindow_henn(w, alg_routing, 0.5);
            case "henn_0.75"://{0,0.25,0.5,0.75,1}
                return new timeWindow_henn(w, alg_routing, 0.75);
            case "henn_1"://{0,0.25,0.5,0.75,1}
                return new timeWindow_henn(w, alg_routing, 1);
            case "zhang_1":
                return new timeWindow_Zhang(w, alg_routing, 1);
            case "zhang_2":
                return new timeWindow_Zhang(w, alg_routing, 2);
            case "zhang_3":
                return new timeWindow_Zhang(w, alg_routing, 3);
            case "fix_arrivePicker":
                return new timeWindow_fix_arrivePicker(w, alg_routing);
            case "fix_batches_1":
                return new timeWindow_fix_batches(w, alg_routing, 1);
            case "fix_batches_2":
                return new timeWindow_fix_batches(w, alg_routing, 2);
            case "fix_batches_3":
                return new timeWindow_fix_batches(w, alg_routing, 3);
            case "fix_batches_4":
                return new timeWindow_fix_batches(w, alg_routing, 4);
            case "fix_batches_5":
                return new timeWindow_fix_batches(w, alg_routing, 5);
            case "fix_batches_6":
                return new timeWindow_fix_batches(w, alg_routing, 6);
            case "fix_orders_P4":
                return new timeWindow_fix_orders(w, alg_routing, 4);
            case "fix_orders_P8":
                return new timeWindow_fix_orders(w, alg_routing, 8);
            case "fix_orders_P16":
                return new timeWindow_fix_orders(w, alg_routing, 16);
            case "fix_time_M3":
                return new timeWindow_fix_time(w, alg_routing, 3);
            case "fix_time_M6":
                return new timeWindow_fix_time(w, alg_routing, 6);
            case "fix_time_M12":
                return new timeWindow_fix_time(w, alg_routing, 12);
            case "fix_orders_P5":
                return new timeWindow_fix_orders(w, alg_routing, 5);
            case "fix_orders_P15":
                return new timeWindow_fix_orders(w, alg_routing, 15);
            case "fix_orders_P25":
                return new timeWindow_fix_orders(w, alg_routing, 25);
            case "fix_time_M10":
                return new timeWindow_fix_time(w, alg_routing, 10);
            case "fix_time_M20":
                return new timeWindow_fix_time(w, alg_routing, 20);
            case "fix_time_M30":
                return new timeWindow_fix_time(w, alg_routing, 30);
            case "random":
                return new timeWindow_random(w, alg_routing);
            case "probability_start":
                return new timeWindow_probability_start(w, alg_routing, threshold);
            case "probability_start2":
                return new timeWindow_probability_start2(w, alg_routing, threshold);
            case "probability_start_new":
                return new timeWindow_probability_start_new(w, alg_routing);
            case "probability_start_peso":
                return new timeWindow_probability_start_peso(w, alg_routing);
            case "probability_start2_peso":
                return new timeWindow_probability_start2_peso(w, alg_routing, threshold);
            case "probability_histograma_peso_P20":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.2);
            case "probability_histograma_peso_P30":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.3);
            case "probability_histograma_peso_P40":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.4);
            case "probability_histograma_peso_P50":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.5);
            case "probability_histograma_peso_P60":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.6);
            case "probability_histograma_peso_P70":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.7);
            case "probability_histograma_peso_P80":
                return new timeWindow_probability_histogram_peso(w, alg_routing, 0.8);
            case "probability_histograma_global_P10":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.1);
            case "probability_histograma_global_P20":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.2);
            case "probability_histograma_global_P30":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.3);
            case "probability_histograma_global_P40":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.4);
            case "probability_histograma_global_P50":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.5);
            case "probability_histograma_global_P60":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.6);
            case "probability_histograma_global_P70":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.7);
            case "probability_histograma_global_P80":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.8);
            case "probability_histograma_global_P90":
                return new timeWindow_probability_histogram_global(w, alg_routing, 0.9);
            case "probability_start_pesoXtiempo":
                return new timeWindow_probability_start_pesoXtiempo(w, alg_routing);
            case "probability_start_pesoXmejoras":
                return new timeWindow_probability_start_pesoXmejoras(w, alg_routing);
            case "probability_start_tiempo":
                return new timeWindow_probability_start_tiempo(w, alg_routing);
            case "probability_start2_tiempo_B1":
                return new timeWindow_probability_start2_tiempo(w, alg_routing, threshold, 1);
            case "probability_start_tiempoXmejoras":
                return new timeWindow_probability_start_tiempoXmejoras(w, alg_routing);
            case "probability_start_mejoras_B1":
                return new timeWindow_probability_start_mejoras(w, alg_routing, threshold, 1);
            case "probability_start_mejoras_reset":
                return new timeWindow_probability_start_mejoras_reset(w, alg_routing, threshold);
            default:
                return new timeWindow_fix_arrivePicker(w, alg_routing);
        }
    }
