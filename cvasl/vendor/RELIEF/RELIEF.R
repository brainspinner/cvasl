#' @export
frob=function(X){ sum(X^2,na.rm=T) }

#' @export
sigma.rmt=function(X){ estim_sigma(X,method="MAD") }

#' @export
softSVD=function(X, lambda){
  svdX=svd(X)
  nuc=pmax(svdX$d-lambda,0)
  out=tcrossprod(svdX$u, tcrossprod( svdX$v,diag(nuc) ))
  return(list(out=out, nuc=sum(nuc)))
}

#' @export
relief=function(dat, batch=NULL, mod=NULL,
                scale.features=T, eps=1e-3, max.iter=1000, verbose=T){
  if (verbose) {
    if (!is.null(mod)){
      q=ncol(mod)
      cat(paste0("[RELIEF] Performing RELIEF harmonization with ", ncol(mod), " covariates\n"))
    }
    else{
      q=0
      cat(paste0("[RELIEF] Performing RELIEF harmonization without covariates\n"))
    }
  }
  if (is.null(batch)){ stop("batch information must be provided\n") }
  p=nrow(dat); n=ncol(dat);
  dat.original=dat
  batch.f=as.factor(batch); batch = as.numeric(batch.f)
  batch.id=unique(batch); n.batch=length(batch.id);batch.f.id=unique(batch.f);
  if (verbose) cat(paste0("[RELIEF] ",n.batch," batch identified\n"))

  Xbeta=gamma=sigma.mat=Matrix(0, p, n)
  batch.covariates=model.matrix(~batch.f-1)

  if (is.null(mod)){
    Xbeta = tcrossprod(apply(dat, 1, mean), rep(1,n))
  }else{
    Px= mod%*%ginv(crossprod(mod), tol=0)%*%t(mod)
    Xbeta= dat%*%Px
  }
  residual1 = dat-Xbeta
  Pb = batch.covariates%*%ginv(crossprod(batch.covariates),tol=0)%*%t(batch.covariates)
  gamma = residual1%*%Pb
  residual2 = residual1-gamma

  if (scale.features){
    sigma.mat=sqrt(rowSums(residual2^2)/(n-n.batch-q))%*%t(rep(1,n))
  } else {
    sigma.mat=1
  }

  dat=residual2/sigma.mat

  sub.batch = unlist(lapply(c(1,n.batch), combn, x = batch.id, simplify = FALSE), recursive = FALSE)

  nvec=rep(NA, n.batch)

  sigma.mat.batch=Matrix(1, p, n)
  for (b in 1:n.batch){
    order.temp.batch=which(batch==batch.id[b])
    nvec[b]=length(order.temp.batch)

    s=sigma.rmt(dat[, order.temp.batch])
    sigma.mat.batch[, order.temp.batch]=sigma.mat.batch[, order.temp.batch]*s
    dat[, order.temp.batch]=dat[, order.temp.batch]/s
  }

  sigma.harnomized=sqrt(sum((unique(as.numeric(sigma.mat.batch))^2)*nvec)/(sum(nvec)))

  lambda.set=matrix(NA, 1,length(sub.batch))
  for (b in 1:length(sub.batch)){
    lambda.set[1,b]=sqrt(p)+sqrt(sum(nvec[sub.batch[[b]]]))
  }

  index.set.batch = lapply(sub.batch, function(b) which(batch %in% b))

  estim = lapply(1:length(sub.batch), function(x) Matrix(0, p, n, sparse = TRUE))

  bool=TRUE
  count=1; crit0=0
  idx=c(1:length(sub.batch))

  if (verbose) {
    cat(paste0("[RELIEF] Start optimizing...\n"))
    pb = txtProgressBar(min = 0, max=max.iter, initial=0, char="-", style = 3)
  }
  while (bool){
    if (verbose){  setTxtProgressBar(pb, count)  }
    crit0.old = crit0
    nuc.temp=matrix(NA,1, length(sub.batch))
    for (b in length(sub.batch):1){
      temp=softSVD( (dat-Reduce("+", estim[-idx[b]]))[,index.set.batch[[b]]],lambda.set[,b])
      estim[[b]][,index.set.batch[[b]]]=temp$out
      nuc.temp[,b]=temp$nuc
    }

    crit0 = 1/2*frob(dat-Reduce("+", estim))+sum(lambda.set*nuc.temp,na.rm=T)
    if (abs(crit0.old-crit0)<eps){ bool=FALSE }
    else if (count==max.iter){ bool=FALSE}
    else{ count = count+1 }
  }

  if (verbose & count<max.iter){
    setTxtProgressBar(pb, max.iter)
    cat(paste0("\n[RELIEF] Convergence reached. Finish harmonizing.\n"))
  }
  if (verbose & count==max.iter){
    cat(paste0("\n[RELIEF] Convergence not reached. Increase max.iter.\n"))
  }

  E=dat-Reduce("+", estim)
  E.scaled=sigma.mat*E
  E.original=sigma.mat*sigma.mat.batch*E
  R=sigma.mat*sigma.mat.batch*estim[[length(index.set.batch)]]
  I=sigma.mat*sigma.mat.batch*Reduce("+", estim[-length(index.set.batch)])
  harmonized=Xbeta+R+sigma.harnomized*E.scaled
  estimates=list(Xbeta=Xbeta,gamma=gamma,sigma.mat=sigma.mat, sigma.mat.batch=as.matrix(sigma.mat.batch),sigma.harnomized=sigma.harnomized, R=as.matrix(R),I=as.matrix(I),E.scaled=as.matrix(E.scaled), E.original=as.matrix(E.original))

  return(list(dat.relief=as.matrix(harmonized),
              estimates=estimates,dat.original=dat.original,
              batch=batch.f))
}
