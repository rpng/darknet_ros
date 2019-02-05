#include "darknet.h"
#include "list.h"
#include "network.h"
#include "image.h"
#include "utils.h"

#include <sys/time.h>
#include <assert.h>
#include <dirent.h>
#include <string.h>
void train_unet_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred = get_network_image(*net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/unet/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;

    char *train_label_list = option_find_str(options, "label", "data/unet/labels.list");

    list *pplist = get_paths(train_label_list);
    char **labels = (char **)list_to_array(pplist);    
    clock_t time;

    data train;
//    load_thread = load_data(args);

//   for (i = 0; i < 215; i++){
//        printf("sample image %f\n",(float)train.y.vals[0][i]);
//    }
    int epoch = (*net->seen)/N;
    while(get_current_batch(*net) < net->max_batches || net->max_batches == 0){
        time=clock();
        train = load_data_unet(paths, imgs, N, labels, net->w, net->h);
//        pthread_join(load_thread, 0);
	paths = (char **)list_to_array(plist);
	labels = (char **)list_to_array(pplist); 
//        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = 0;

#ifdef GPU
        if(ngpus == 1){
            loss = train_network(*net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);

#endif
        if(display){
            image tr = float_to_image(net->w/div, net->h/div, 80, train.y.vals[net->batch*(net->subdivisions-1)]);
            image im = float_to_image(net->w, net->h, net->c, train.X.vals[net->batch*(net->subdivisions-1)]);
            image mask = mask_to_rgb(tr);
            image prmask =mask_to_rgb(pred);
            show_image(im, "input");
            //show_image(prmask, "pred");
            //show_image(mask, "truth");
#ifdef OPENCV
            cvWaitKey(100000);
#endif
            free_image(mask);
            free_image(prmask);
        }
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(*net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(*net), sec(clock()-time), *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(*net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(*net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void predict_unet_segmenter()
{
    srand(2222222);
    DIR *dir;
    struct dirent *ent;
    char *cfg = "./unet.cfg";
    char *weights = "./unet.backup"; // You have load your model here

    char dirname[256],resdirname[256], filename[256], resfilename[256];
    strcpy(dirname,"./data/unet/test/");
    strcpy(resdirname,"./data/unet/result/");
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    if ((dir = opendir (dirname)) != NULL) {
        clock_t time;
        char buff[256];
        char *input = buff;
        while ((ent = readdir (dir)) != NULL) {
            if (strstr(ent->d_name, "png")!= NULL) {
                strcpy(filename, dirname);
                strcat(filename, ent->d_name);
                strcpy(resfilename, resdirname);
                strcat(resfilename, ent->d_name);                
                printf ("%s\n", filename);
                strncpy(input, filename, 256);
                image im = load_image_color(input, 0, 0);
                float *X = (float *) im.data;
                time=clock();
                float *predictions = network_predict(*net, X);
                image pred = get_network_image(*net);
                image prmask = mask_to_rgb(pred);
		save_image_png(prmask, resfilename);
                show_image(prmask, "orig");
		cvWaitKey(1000000);
                //printf("Predicted: %f\n", predictions[0]);
                printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		free_image(prmask);
        }
        }
        closedir (dir);                
    } else{
        /* could not open directory */
        perror ("");
    }   
}

void run_unet_segmenter(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) predict_unet_segmenter();
    else if(0==strcmp(argv[2], "train")) train_unet_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
}


