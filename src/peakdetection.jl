
function findpeaks(activation; filterlength=10, minh=0, threshold=-Inf, minw=0, maxw=Inf, doplot=false, kwargs...)
    activationf = filtfilt(ones(filterlength),[filterlength],activation)
    doplot && plot!(activationf, lab="Filtered Activation", kwargs...)

    peaks = allpeaks(activationf)
    doplot &&  scatter!(peaks, activationf[peaks], lab="All peaks", kwargs...)

    peaks = peaks[activationf[peaks] .>= minh] # Remove too small peaks
    doplot &&  scatter!(peaks, activationf[peaks]*1.1, lab="Below minh removed", kwargs...)

    peaks = remove_threshold(peaks,activationf,threshold)
    peaks = remove_width(peaks,activationf,minw,maxw)
    doplot &&  scatter!(peaks, activationf[peaks]*1.2, lab="Too wide or narrow removed" , m=(10,:xcross), kwargs...)
    return peaks
end

function remove_threshold(peaks,activation,threshold)
    base = max.(activation[peaks .- 1],activation[peaks .+ 1])
    peaks = peaks[activation[peaks].-base .>= threshold]
end

function remove_width(peaks,activation,minw,maxw)
    if minw <= 0 && maxw == Inf
        return peaks # Nothing needs to be done
    end
    i = 1
    while i < length(peaks)
        speaks = sortperm(activation[peaks], rev=true) # sorted peak locations (in peaks vector)

        if speaks[i] > 1
            wl = abs(peaks[speaks[i]] - peaks[speaks[i]-1]) # highest peak location - location to the left
            if wl < minw || wl > maxw
                deleteat!(peaks,speaks[i]-1)
                continue
            end
        end
        if speaks[i] < length(peaks)
            wr = abs(peaks[speaks[i]] - peaks[speaks[i]+1])
            if wr < minw || wr > maxw
                deleteat!(peaks,speaks[i]+1)
                continue
            end
        end
        i += 1
    end
    if !isempty(peaks)
        @assert minimum(diff(peaks)) >= minw
    end
    peaks
end

function allpeaks(activation)
    da = diff(activation)
    peaks = findall((diff(da) .< 0) .& (da[1:end-1] .> 0) .& (da[2:end] .< 0)) .+ 1
end
